//! `nuclis model`: acquiring artifacts through the `huggingface` package,
//! listing what the models directory holds, removing an entry's files
//! (`rm`), and judging a remote file from its directory alone before any
//! download (`inspect`).
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
/// `support` is a safetensors artifact's configuration, tokenizer, or index;
/// `codec` a speech model's codec weights, pulled with it.
pub const Role = enum {
    main,
    mmproj,
    draft,
    imatrix,
    support,
    codec,
    backbone,

    /// A sidecar written before draft variants says `mtp` for a draft.
    pub fn jsonParse(alloc: Allocator, source: anytype, options: std.json.ParseOptions) !Role {
        const token = try source.nextAllocMax(alloc, .alloc_if_needed, options.max_value_len.?);
        const text = switch (token) {
            inline .string, .allocated_string => |t| t,
            else => return error.UnexpectedToken,
        };
        return parse(text) orelse error.InvalidEnumTag;
    }

    /// A role by name, `mtp` included as the old spelling of `draft`.
    pub fn parse(text: []const u8) ?Role {
        if (std.mem.eql(u8, text, "mtp")) return .draft;
        return std.meta.stringToEnum(Role, text);
    }
};

pub const sidecar_suffix = ".nuclis.json";
/// A sidecar is a few hundred bytes; anything larger is not ours.
pub const max_sidecar_bytes = 64 * 1024;
/// Upper bound on files a listing walks, and on nesting below a repository.
pub const max_listed_files = 10_000;
pub const max_depth_below_repo = 4;

pub const PullOptions = struct {
    /// An index name (`qwen3.8-27b`) or a repository id (`owner/repo`).
    repo: []const u8 = "",
    /// Companions to add to an index pull, by role; `all` takes each role's
    /// default file, and every draft variant with `variants`. Meaningless
    /// with a repository id.
    with: std.EnumSet(Role) = .empty,
    /// `--with draft=<variant>`: that draft instead of the default.
    draft_variant: ?[]const u8 = null,
    all: bool = false,
    variants: bool = false,
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
    draft: ?[]const u8 = null,
    /// `--register <name>`: write the pulled files as a registry entry of
    /// `config_path` once they are verified; `profile` forces the entry's
    /// prompt profile (`--profile`, meaningful only with `--register`).
    register: ?[]const u8 = null,
    profile: ?config.Profile = null,
    config_path: ?[]const u8 = null,
};

/// A name that is neither an index name nor an `owner/repo` id can only
/// be a registry entry: the caller then reads `nuclis.json`, which the
/// other forms never need (a broken file must not block a download).
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
    if (options.draft_variant != null) {
        diag.set("{s} is a registry entry: it names one draft file; --with draft=<variant> needs an index name", .{options.repo});
        return error.ConflictingOptions;
    }
    var wanted = options.with.iterator();
    while (wanted.next()) |role| {
        const companion: ?[]const u8 = switch (role) {
            .mmproj => entry.mmproj,
            .draft => entry.draft,
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
    translated.with = .empty;
    translated.all = false;
    translated.variants = false;
    if (options.all or options.with.contains(.mmproj)) translated.mmproj = entry.mmproj;
    if (options.all or options.with.contains(.draft)) translated.draft = entry.draft;
    return translated;
}

/// The provenance record. `revision` is the 40-character commit and
/// `sha256` the 64-character lowercase digest, both as the Hub reported
/// them; `size` lets `ls` flag a file that changed underneath its sidecar.
/// Schema 2 adds the artifact and variant a file belongs to and `model
/// adopt`'s record: `adopted_at` instead of `downloaded_at`, and no
/// `revision` when the file matched no pinned file (its digest is then the
/// local one, which keeps the file identity-stable). Schema 1 still reads.
pub const Sidecar = struct {
    schema_version: u32 = 2,
    repo: []const u8,
    file: []const u8,
    /// Null for an adopted file that matched no pinned one.
    revision: ?[]const u8 = null,
    sha256: []const u8,
    size: u64,
    role: Role,
    /// The index artifact the file belongs to, and its draft variant.
    artifact: ?[]const u8 = null,
    variant: ?[]const u8 = null,
    /// RFC 3339 UTC, second resolution: one of the two is set.
    downloaded_at: ?[]const u8 = null,
    adopted_at: ?[]const u8 = null,
    /// The Hub's git blob id (40 hex) for a plain git file, which has no
    /// Hub SHA-256: `sha256` is then computed locally after the blob id
    /// verified the bytes. Null for LFS files.
    git_blob: ?[]const u8 = null,
    nuclis_version: []const u8,

    /// The commit's first 12 characters, or a dash for an unmatched adoption.
    pub fn shortRevision(self: Sidecar) []const u8 {
        return if (self.revision) |r| r[0..12] else "-";
    }
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
    if (parsed.schema_version < 1 or parsed.schema_version > 2 or parsed.sha256.len != 64) return error.InvalidSidecar;
    if (parsed.revision) |r| if (r.len != 40) return error.InvalidSidecar;
    if (parsed.schema_version == 1 and (parsed.revision == null or parsed.downloaded_at == null)) return error.InvalidSidecar;
    return parsed;
}

/// Written whole; a sidecar is small enough that a torn write is a
/// `InvalidSidecar` on the next read, which `pull` treats as absent.
pub fn writeSidecar(alloc: Allocator, io: std.Io, dir: std.Io.Dir, sub_path: []const u8, sidecar: Sidecar) !void {
    var buffer: std.Io.Writer.Allocating = .init(alloc);
    defer buffer.deinit();
    try std.json.Stringify.value(sidecar, .{ .whitespace = .indent_2, .emit_null_optional_fields = false }, &buffer.writer);
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
/// Null when it says nothing, or when this build cannot read the directory
/// (an encoding it does not store, a shape it rejects): the download is
/// digest-verified either way, so the sidecar is written with the request's
/// role rather than the file left looking unverified.
fn roleFromHeader(alloc: Allocator, io: std.Io, path: []const u8) !?Role {
    var doc = inference.gguf.open(alloc, io, path, .{}) catch |err| switch (err) {
        error.UnsupportedTensorType, error.InvalidShape, error.Overflow => return null,
        else => return err,
    };
    defer doc.deinit();
    if (doc.string("general.type")) |kind| {
        if (std.mem.eql(u8, kind, "imatrix")) return .imatrix;
        if (std.mem.eql(u8, kind, "mmproj")) return .mmproj;
    }
    if (doc.string("general.architecture")) |arch| if (std.mem.eql(u8, arch, "clip")) return .mmproj;
    return null;
}

/// Whether the GGUF header names the architecture `nuclis embed` runs.
fn isEmbeddingFile(alloc: Allocator, io: std.Io, path: []const u8) !bool {
    var doc = inference.gguf.open(alloc, io, path, .{}) catch |err| switch (err) {
        error.UnsupportedTensorType, error.InvalidShape, error.Overflow => return false,
        else => return err,
    };
    defer doc.deinit();
    return std.mem.eql(u8, doc.string("general.architecture") orelse "", inference.embed.architecture);
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
    /// Bytes continued from a partial download an earlier pull kept.
    resumed_bytes: u64 = 0,
    role: Role,
    sidecar: []const u8,
};

const PullReport = struct {
    schema_version: u32 = 3,
    /// The catalogue or registry name when one was given, else null.
    name: ?[]const u8,
    repo: []const u8,
    requested_revision: []const u8,
    revision: []const u8,
    files: []const PulledFile,
    /// The registry entry `--register` wrote, the file it lives in, and the
    /// key that makes it the default.
    registered: ?struct { name: []const u8, config: []const u8, key: []const u8 = "engine.model" } = null,
};

const SelectionReport = struct {
    schema_version: u32 = 1,
    repo: []const u8,
    revision: []const u8,
    selection_required: []const Choice,
    const Choice = struct { name: []const u8, size: u64 };
};

/// One file to fetch: its exact Hub name, the role the caller knows (the
/// index's or `--role`), and the digest it must have (the index's pinned
/// one, or the Hub's own for a raw repository): a SHA-256, or for a plain
/// git file only its blob id. `pinned` marks a digest the index states;
/// `destination` places the file away from its repository's layout.
const Job = struct {
    name: []const u8,
    role: ?Role,
    sha256: ?[]const u8,
    git_blob: ?[]const u8 = null,
    size: u64,
    pinned: bool = false,
    variant: ?[]const u8 = null,
    destination: ?[]const u8 = null,
};

fn hexOrNull(arena: Allocator, bytes: anytype) !?[]const u8 {
    const value = bytes orelse return null;
    return try arena.dupe(u8, &std.fmt.bytesToHex(value, .lower));
}

fn remoteJob(arena: Allocator, f: hf.RemoteFile, role: ?Role) !Job {
    const kind = hf.format(f.name);
    return .{
        .name = try arena.dupe(u8, f.name),
        .role = if (kind == null) .support else role,
        .sha256 = try hexOrNull(arena, f.sha256),
        .git_blob = try hexOrNull(arena, f.git_oid),
        .size = f.size,
    };
}

/// The files of an index artifact that come from one repository at one
/// commit: one listing and one report.
const Plan = struct {
    artifact: *const catalog.Artifact,
    repo: []const u8,
    revision: []const u8,
    files: []const *const catalog.File,
};

/// Which of an artifact's files a pull takes: the main file and those
/// pulled with it, then `--with` roles (a draft by variant or the
/// default), `--all` (each role's default, every variant with
/// `--variants`). Refuses a role or variant the artifact does not have.
pub fn selectFiles(arena: Allocator, a: *const catalog.Artifact, options: PullOptions, diag: *config.Diagnostic) ![]const *const catalog.File {
    var wanted = options.with.iterator();
    while (wanted.next()) |role| if (a.companion(role) == null) {
        diag.set("{s} has no {s} companion in the index", .{ a.name, @tagName(role) });
        return error.NoSuchCompanion;
    };
    if (options.draft_variant) |v| if (a.draft(v) == null) {
        var names: std.ArrayList(u8) = .empty;
        for (a.files) |f| if (f.role == .draft) if (f.variant) |fv| {
            if (names.items.len > 0) try names.appendSlice(arena, ", ");
            try names.appendSlice(arena, fv);
        };
        diag.set("{s} has no draft variant {s} (it has: {s})", .{ a.name, v, if (names.items.len > 0) names.items else "a single unnamed draft" });
        return error.NoSuchCompanion;
    };
    var files: std.ArrayList(*const catalog.File) = .empty;
    for (a.files) |*f| {
        const take = catalog.pulledWith(f.role) or switch (f.role) {
            .draft => if (options.all and options.variants) true else if (options.draft_variant) |v| std.mem.eql(u8, f.variant orelse "", v) else (options.all or options.with.contains(.draft)) and f == a.draft(null).?,
            else => options.all or options.with.contains(f.role),
        };
        if (take) try files.append(arena, f);
    }
    return files.items;
}

/// Resolves the repository (an index name or an `owner/repo` id) and its
/// revision, prints the pinned commit, downloads or verifies each file,
/// and writes the sidecars. An index artifact whose files come from
/// several repositories pulls one repository at a time, each with its own
/// report. `diag` carries the reason of every typed failure.
pub fn pull(gpa: Allocator, io: std.Io, environ: *const std.process.Environ.Map, root: []const u8, options: PullOptions, json: bool, out: *std.Io.Writer, sty: style.Style, diag: *config.Diagnostic) !void {
    const a = catalog.find(options.repo) orelse return pullOne(gpa, io, environ, root, options, null, json, out, sty, diag);
    if (options.file != null or options.revision != null) {
        diag.set("{s} is an index name and pins its files and commit; pull by repository id ({s}) to choose another", .{ a.name, a.repo });
        return error.ConflictingOptions;
    }
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const groups = try groupByOrigin(arena, a, try selectFiles(arena, a, options, diag));
    if (options.register != null and groups.len > 1) {
        diag.set("{s} pulls from {d} repositories; register it with `nuclis config sync` after the pull", .{ a.name, groups.len });
        return error.ConflictingOptions;
    }
    // The other repositories first, so the artifact's own report is last.
    for (groups[1..]) |g| try pullOne(gpa, io, environ, root, options, g, json, out, sty, diag);
    return pullOne(gpa, io, environ, root, options, groups[0], json, out, sty, diag);
}

/// `files` as one plan per origin (repository and commit), in first-seen
/// order: the main file's comes first.
pub fn groupByOrigin(arena: Allocator, a: *const catalog.Artifact, files: []const *const catalog.File) ![]const Plan {
    var groups: std.ArrayList(Plan) = .empty;
    for (files) |f| {
        const repo = a.origin(f);
        const revision = a.originRevision(f);
        const group = for (groups.items) |*g| {
            if (std.mem.eql(u8, g.repo, repo) and std.mem.eql(u8, g.revision, revision)) break g;
        } else null;
        if (group) |g| {
            g.files = try std.mem.concat(arena, *const catalog.File, &.{ g.files, &.{f} });
        } else try groups.append(arena, .{ .artifact = a, .repo = repo, .revision = revision, .files = try arena.dupe(*const catalog.File, &.{f}) });
    }
    return groups.items;
}

fn pullOne(gpa: Allocator, io: std.Io, environ: *const std.process.Environ.Map, root: []const u8, options: PullOptions, plan: ?Plan, json: bool, out: *std.Io.Writer, sty: style.Style, diag: *config.Diagnostic) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    if (options.repo.len == 0) return error.MissingRepository;
    const models = try modelsDir(arena, root);

    interrupt.install();
    interrupt.installTerminate();
    // The package's allocator must be thread safe: the concurrent workers
    // share it. `gpa` from `std.process.Init` is.
    var client = hf.Client.init(gpa, io, environ);
    var display: Display = .{ .io = io, .started = .now(io, .awake), .tty = std.Io.File.stderr().isTty(io) catch false, .json = json };
    client.progress = .{ .context = &display, .update = Display.update };

    // 1. Resolve. An index plan pins repository, files, commit, and digests
    //    from the table; a repository id resolves them on the Hub.
    var jobs: std.ArrayList(Job) = .empty;
    var request: hf.Request = .{ .repo_id = options.repo, .revision = options.revision orelse "main", .local_dir = models };
    var pinned: [40]u8 = undefined;
    if (plan) |p| {
        const a = p.artifact;
        request.repo_id = p.repo;
        request.revision = p.revision;
        @memcpy(&pinned, p.revision[0..40]);
        // Safetensors weights and a codec bring support files: one listing
        // at the pinned commit names them, and must hold every one the
        // index names.
        var listed: ?hf.Catalog = null;
        defer if (listed) |*l| l.deinit();
        for (p.files) |f| if (f.support.len > 0 or hf.format(f.path) == .safetensors) {
            listed = client.list(request) catch |err| return listFailure(err, arena, client, request, options.repo, diag);
            break;
        };
        for (p.files) |f| {
            const name = catalog.Artifact.hubName(f);
            const destination: ?[]const u8 = if (f.remote != null) try catalog.localPath(arena, models, a, f) else null;
            const first = jobs.items.len;
            if (listed) |*remote| if (f.support.len > 0 or hf.format(f.path) == .safetensors) {
                const selected = (hf.select(arena, remote, name) catch |err| return hubFailure(err, options.repo, diag)).?;
                for (selected) |r| {
                    var job = try remoteJob(arena, r, if (std.mem.eql(u8, r.name, name)) f.role else null);
                    if (std.mem.eql(u8, r.name, name)) {
                        if (!std.mem.eql(u8, job.sha256 orelse "", f.sha256)) {
                            diag.set("{s}: the Hub's digest at commit {s} is not the index's {s}…; the index entry needs updating", .{ name, p.revision[0..12], f.sha256[0..12] });
                            return error.CatalogMismatch;
                        }
                        job.pinned = true;
                    }
                    try jobs.append(arena, job);
                }
                for (f.support) |support| {
                    for (jobs.items[first..]) |job| {
                        if (std.mem.eql(u8, job.name, support)) break;
                    } else {
                        diag.set("{s}: {s} is not beside the weights at commit {s}; the index entry needs updating", .{ a.name, support, p.revision[0..12] });
                        return error.CatalogMismatch;
                    }
                }
                continue;
            };
            try jobs.append(arena, .{ .name = name, .role = f.role, .sha256 = f.sha256, .size = f.size, .pinned = true, .variant = f.variant, .destination = destination });
        }
        if (listed) |l| pinned = l.revision;
    } else {
        if (options.all or options.with.count() != 0 or options.draft_variant != null) {
            diag.set("--with and --all need an index name; {s} is a repository id (pass --file per companion)", .{options.repo});
            return error.ConflictingOptions;
        }
        request.filename = options.file;
        var remote = client.list(request) catch |err| return listFailure(err, arena, client, request, options.repo, diag);
        defer remote.deinit();
        pinned = remote.revision;
        const selected = (hf.select(arena, &remote, request.filename) catch |err| return hubFailure(err, options.repo, diag)) orelse {
            try renderSelection(out, .{ .repo = options.repo, .revision = &pinned, .selection_required = try choices(arena, remote.files) }, json, sty);
            try out.flush();
            diag.set("{s} has {d} model files; pass --file <name>", .{ options.repo, remote.files.len });
            return error.SelectionRequired;
        };
        for (selected) |f| try jobs.append(arena, try remoteJob(arena, f, options.role));
        // A registry entry's companions, by exact name at the same commit.
        for ([_]struct { name: ?[]const u8, role: Role }{ .{ .name = options.mmproj, .role = .mmproj }, .{ .name = options.draft, .role = .draft } }) |companion| {
            const name = companion.name orelse continue;
            const file = for (remote.files) |f| {
                if (std.mem.eql(u8, f.name, name)) break f;
            } else {
                diag.set("{s}: no GGUF named {s} in the repository (the registry entry's {s} companion)", .{ options.repo, name, @tagName(companion.role) });
                return error.FileNotFound;
            };
            try jobs.append(arena, try remoteJob(arena, file, companion.role));
        }
    }
    const revision: []const u8 = &pinned;
    request.revision = revision;
    var model_kind: ?catalog.ModelKind = if (plan) |p| if (p.artifact.kind != .generation) p.artifact.kind else null else null;
    if (options.register) |name| {
        if (options.config_path == null) return error.MissingHome;
        const main_file = for (jobs.items) |job| {
            if (job.role == null or job.role == .main) break job.name;
        } else null;
        if (plan == null) if (main_file) |f| if (hf.format(f) == .safetensors) {
            // Laya's layout registers as a decision model; any other
            // checkpoint only once its downloaded config.json proves it a
            // speech model (`speechCheckpoint`, after the transfer).
            if (besideWeights(f, jobs.items, "rl_agent_config.json")) {
                model_kind = .decision;
            } else if (!besideWeights(f, jobs.items, "config.json")) {
                diag.set("{s} is a safetensors artifact with neither rl_agent_config.json nor config.json beside it; only a Laya decision or a Qwen3-TTS CustomVoice checkpoint registers (pull it without --register)", .{f});
                return error.NotRunnable;
            }
        };
        try config.registrable(name, request.repo_id, main_file, diag);
    }
    const pinned_name: ?[]const u8 = if (plan) |p| p.artifact.name else null;
    if (!json) try renderHeader(out, sty, pinned_name orelse options.name, request.repo_id, if (pinned_name != null) null else options.revision orelse "main", revision);
    try out.flush();

    // 2. Destinations: a sidecar recording other content is a conflict
    //    unless `--force`, whose verified download replaces the file (the
    //    old one stays until then) and then its sidecar; a file without one
    //    is left to the package's verification.
    for (jobs.items) |job| {
        var placed = request;
        placed.destination = job.destination;
        placed.exact = job.destination != null;
        const path = try client.localPath(placed, job.name);
        defer gpa.free(path);
        const sidecar = try sidecarPath(arena, path);
        const recorded = readSidecar(arena, io, .cwd(), sidecar) catch |err| switch (err) {
            error.InvalidSidecar => null,
            else => return err,
        };
        if (recorded) |r| {
            // A plain git file has only its blob id on the Hub.
            const same = if (job.sha256) |sha| std.mem.eql(u8, r.sha256, sha) else std.mem.eql(u8, r.git_blob orelse "", job.git_blob.?);
            if (same and r.size == job.size) continue;
            if (!options.force) {
                const want, const kind = if (job.sha256) |sha| .{ sha, "sha256" } else .{ job.git_blob.?, "git blob" };
                const have = if (job.sha256 != null) r.sha256 else r.git_blob orelse "(none recorded)";
                diag.set("{s}: its sidecar records commit {s} ({s} {s}…), the request resolved to commit {s} ({s} {s}…); pass --force to replace it", .{ path, r.shortRevision(), kind, have[0..@min(12, have.len)], revision[0..12], kind, want[0..12] });
                return error.ExistingFileMismatch;
            }
        }
    }

    // 3. Transfer, one package call per job (a split file expands to its
    //    shard set inside the package). The package verifies existing
    //    files, overwrites only under `--force` (`replace`), removes its
    //    partial file on failure and Ctrl-C, and keeps it on SIGTERM for
    //    the next pull to continue. 4. Provenance, only once a file is
    //    verified, so an interrupted multi-file pull keeps the sidecars of
    //    what it finished.
    request.replace = options.force;
    var report: std.ArrayList(PulledFile) = .empty;
    // What `--register` records: filled per verified file by its role.
    var registration: config.Registration = .{ .repo = request.repo_id, .revision = revision, .file = null, .profile = options.profile, .kind = model_kind };
    var main_path: ?[]const u8 = null;
    var stamp: [20]u8 = undefined;
    const downloaded_at = rfc3339(&stamp, std.Io.Timestamp.now(io, .real).toSeconds());
    // Every job is one file (shard sets and support files were expanded
    // above), so each download is exact.
    request.exact = true;
    display.count = jobs.items.len;
    for (jobs.items) |job| display.total += job.size;
    for (jobs.items, 0..) |job, index| {
        request.filename = job.name;
        request.destination = job.destination;
        display.index = index;
        display.file = job.name;
        display.file_size = job.size;
        defer display.done += job.size;
        var result = client.download(request) catch |err| switch (err) {
            error.Canceled => {
                diag.set("interrupted; the partial download was removed", .{});
                return error.Cancelled;
            },
            error.Paused => {
                diag.set("stopped; the partial download of {s} was kept, and pulling again continues it", .{job.name});
                return error.Cancelled;
            },
            error.ResumedChecksumMismatch => {
                diag.set("{s}: the partial download kept from an earlier pull did not verify and was removed; pull again to download it whole", .{job.name});
                return err;
            },
            error.DownloadInProgress => {
                diag.set("{s}: another pull is downloading this file", .{job.name});
                return err;
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
            const sidecar = try sidecarPath(arena, file.path);
            // The Hub at the pinned commit must agree with the catalogue;
            // otherwise the table is wrong, and the file stays without a
            // sidecar so it is never taken for verified (a replaced file's
            // old sidecar goes too).
            if (job.pinned and !std.mem.eql(u8, digest, job.sha256.?)) {
                if (options.force) try deleteIfPresent(io, sidecar);
                diag.set("{s}: the Hub's digest at commit {s} is {s}…, the index pins {s}…; the index entry needs updating", .{ file.path, revision[0..12], digest[0..12], job.sha256.?[0..12] });
                return error.CatalogMismatch;
            }
            const header = if (hf.format(file.path) == .gguf) try roleFromHeader(gpa, io, file.path) else null;
            const role = resolveRole(header, job.role) catch |err| {
                if (options.force) try deleteIfPresent(io, sidecar);
                diag.set("{s}: the header says its role is {s}; --role {s} contradicts it (no sidecar written)", .{ file.path, @tagName(header.?), @tagName(job.role.?) });
                return err;
            };
            // The Hub name of a published file (a shard set returns
            // several): the path below the repository directory, without
            // the separator `join` leaves in front of it.
            const repo_dir = try std.fs.path.join(arena, &.{ models, request.repo_id });
            try writeSidecar(arena, io, .cwd(), sidecar, .{
                .repo = request.repo_id,
                .file = if (job.destination == null and std.mem.startsWith(u8, file.path, repo_dir)) std.mem.trimStart(u8, file.path[repo_dir.len..], "/") else job.name,
                .revision = revision,
                .sha256 = digest,
                .size = file.size,
                .role = role,
                .artifact = if (plan) |p| p.artifact.name else null,
                .variant = job.variant,
                .downloaded_at = downloaded_at,
                .nuclis_version = version,
                .git_blob = try hexOrNull(arena, file.remote.git_oid),
            });
            try report.append(arena, .{ .path = try arena.dupe(u8, file.path), .size = file.size, .sha256 = digest, .transport = @tagName(file.transport), .resumed_bytes = file.resumed_bytes, .role = role, .sidecar = sidecar });
            // A repository's GGUF registers as what its header says it is.
            if (role == .main and options.register != null and model_kind == null and hf.format(file.path) == .gguf and try isEmbeddingFile(gpa, io, file.path)) registration.kind = .embedding;
            switch (role) {
                .main => {
                    registration.file = job.name;
                    main_path = try arena.dupe(u8, file.path);
                },
                .mmproj => registration.mmproj = job.name,
                .draft => registration.draft = job.name,
                .imatrix, .support, .codec, .backbone => {},
            }
        }
    }
    if (options.register != null and registration.kind == null) if (main_path) |p| if (hf.format(p) == .safetensors) {
        registration.kind = try speechCheckpoint(gpa, io, p, diag);
    };
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
        const key = switch (registration.kind orelse .generation) {
            .generation => "engine.model",
            .decision => "decide.model",
            .embedding => "embed.model",
            .speech => "speech.model",
            .transcription => "transcribe.model",
        };
        registered = .{ .name = name, .config = config_path, .key = key };
    }
    try renderPull(out, .{ .name = pinned_name orelse options.name, .repo = request.repo_id, .requested_revision = options.revision orelse if (plan) |p| p.revision else "main", .revision = revision, .files = report.items, .registered = registered orelse null }, json, sty);
}

/// Whether the files pulled hold `name` in the weights' own directory.
fn besideWeights(weights: []const u8, jobs: []const Job, name: []const u8) bool {
    const dir = std.fs.path.dirnamePosix(weights) orelse "";
    for (jobs) |job| {
        const job_dir = std.fs.path.dirnamePosix(job.name) orelse "";
        if (std.mem.eql(u8, job_dir, dir) and std.mem.eql(u8, std.fs.path.basenamePosix(job.name), name)) return true;
    }
    return false;
}

/// `.speech` when the configuration beside the pulled `weights` is one
/// `nuclis speak` opens (the same parse it runs); otherwise `NotRunnable`,
/// with the files kept and nothing registered.
fn speechCheckpoint(gpa: Allocator, io: std.Io, weights: []const u8, diag: *config.Diagnostic) !catalog.ModelKind {
    const max_config_bytes = 1024 * 1024;
    const dir = std.fs.path.dirname(weights) orelse ".";
    const config_path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(config_path);
    const generation_path = try std.fs.path.join(gpa, &.{ dir, "generation_config.json" });
    defer gpa.free(generation_path);
    const config_json = try std.Io.Dir.cwd().readFileAlloc(io, config_path, gpa, .limited(max_config_bytes));
    defer gpa.free(config_json);
    const generation_json = std.Io.Dir.cwd().readFileAlloc(io, generation_path, gpa, .limited(max_config_bytes)) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    defer if (generation_json) |g| gpa.free(g);
    var parsed = inference.models.qwen3_tts.parseConfig(gpa, config_json, generation_json) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            diag.set("{s}: {s}; only Laya decision and Qwen3-TTS CustomVoice checkpoints register (the files stay, nothing was registered)", .{ config_path, @errorName(err) });
            return error.NotRunnable;
        },
    };
    parsed.deinit();
    return .speech;
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
    try out.print("{s}{s}{s} @ commit {s}{s}{s} has {s}{d}{s} model files; choose one with --file <name>:\n", .{ sty.on(.code), report.repo, sty.off(), sty.on(.comment), report.revision, sty.off(), sty.on(.number), report.selection_required.len, sty.off() });
    for (report.selection_required) |c| try out.print("  {s}{s}{s}\t{s}{d}{s}\n", .{ sty.on(.code), c.name, sty.off(), sty.on(.number), c.size, sty.off() });
}

fn renderPull(out: *std.Io.Writer, report: PullReport, json: bool, sty: style.Style) !void {
    if (json) {
        try std.json.Stringify.value(report, .{ .whitespace = .indent_2 }, out);
        return out.writeByte('\n');
    }
    for (report.files) |f| {
        try out.print("{s}{s}{s} {s}{s}{s}\n  {s}{d}{s} bytes, sha256 {s}{s}{s}, role {s}{s}{s}; provenance beside it in {s}{s}{s}{s}\n", .{
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
    if (report.registered) |r| try out.print("{s}registered{s} as {s}{s}{s} in {s}{s}{s} {s}(`nuclis config set {s} {s}` makes it the default){s}\n", .{ sty.on(.success), sty.off(), sty.on(.keyword), r.name, sty.off(), sty.on(.code), r.config, sty.off(), sty.on(.dim), r.key, r.name, sty.off() });
}

fn deleteIfPresent(io: std.Io, path: []const u8) !void {
    std.Io.Dir.cwd().deleteFile(io, path) catch |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    };
}

/// `hubFailure`, plus the two refusals of `--file` itself, which name the
/// file: a path outside the repository, and a file that is not a weight,
/// where the repository is listed to name the weights it comes with.
fn listFailure(err: anyerror, arena: Allocator, client: hf.Client, request: hf.Request, repo: []const u8, diag: *config.Diagnostic) anyerror {
    const name = request.filename orelse return hubFailure(err, repo, diag);
    switch (err) {
        error.InvalidFilename => diag.set("{s}: --file takes a path inside the repository (no empty, '.' or '..' parts, '\\', ':', or control characters)", .{name}),
        error.NotAWeightFile => {
            var listing = request;
            listing.filename = null;
            var remote = client.list(listing) catch |list_err| return hubFailure(list_err, repo, diag);
            defer remote.deinit();
            for (remote.support) |f| if (std.mem.eql(u8, f.name, name)) {
                const owned = try hf.owners(arena, &remote, name);
                if (owned.len == 0) {
                    diag.set("{s} is not part of any safetensors artifact in {s}; support files come only with their weights", .{ name, repo });
                } else diag.set("{s} is a support file of {s}: it comes with the weights (pass --file {s})", .{ name, try std.mem.join(arena, ", ", owned), owned[0] });
                return err;
            };
            diag.set("{s}: not a weight file (.gguf, .safetensors) nor a support file nuclis pulls from {s}", .{ name, repo });
        },
        else => return hubFailure(err, repo, diag),
    }
    return err;
}

/// Names the fix for the Hub's refusals (the package's typed errors); other
/// errors pass through unchanged.
fn hubFailure(err: anyerror, repo: []const u8, diag: *config.Diagnostic) anyerror {
    switch (err) {
        error.TokenRequired => diag.set("{s} needs authentication: set HF_TOKEN to a Hugging Face user access token", .{repo}),
        error.TokenRejected => diag.set("HF_TOKEN was refused by the Hub: check the token value and that it has not expired or been revoked", .{}),
        error.AccessDenied => diag.set("access to {s} denied: if it is gated, accept its terms on huggingface.co with the account HF_TOKEN belongs to", .{repo}),
        error.NotFound => diag.set("{s}: no such repository, revision, or file on the Hub", .{repo}),
        error.FileNotFound => diag.set("{s}: no model file of that name in the repository (names match in full, subdirectories included)", .{repo}),
        else => {},
    }
    return err;
}

/// Renders the package's progress on stderr: one line per phase and, on a
/// terminal, a line rewritten in place at most every 200 ms with the file's
/// percentage, bytes, throughput since the download started, and the
/// estimated remaining time. Off a terminal only the phase lines print.
/// With `--json`, one JSON object per line instead (`Progress`; the format
/// is in docs/guide/getting-started.md § Model download). Write failures
/// are ignored: progress is advisory. The interrupt flag is polled here
/// because this is the one callback the package makes between chunks.
const Display = struct {
    io: std.Io,
    started: std.Io.Timestamp,
    tty: bool,
    json: bool = false,
    last_line: ?std.Io.Timestamp = null,
    last_phase: ?@FieldType(hf.ProgressEvent, "phase") = null,
    download_started: ?std.Io.Timestamp = null,
    download_base: u64 = 0,
    open_line: bool = false,
    /// The pull's files, set by `pullOne` before each package call: this
    /// one's Hub name, index, and size, and the bytes of the files before it.
    file: []const u8 = "",
    index: usize = 0,
    count: usize = 0,
    file_size: u64 = 0,
    done: u64 = 0,
    total: u64 = 0,

    fn update(context: ?*anyopaque, event: hf.ProgressEvent) hf.Progress.Error!void {
        const d: *Display = @ptrCast(@alignCast(context.?));
        if (interrupt.requested()) {
            d.render(.{ .phase = .selection_required }) catch {};
            // SIGTERM (an app quitting) keeps the partial; Ctrl-C removes it.
            return if (interrupt.terminated()) error.Paused else error.Canceled;
        }
        d.render(event) catch {};
    }

    /// One `--json` progress line.
    pub const Progress = struct {
        event: []const u8,
        file: []const u8,
        index: usize,
        count: usize,
        /// This file's bytes verified or downloaded, and its size.
        done: u64,
        size: u64,
        /// Every file's of the pull, this one's included.
        pull_done: u64,
        pull_size: u64,
    };

    /// The line for `event` as `--json` prints it; null when none is due (a
    /// download line at most every 250 ms, and at the file's end).
    fn progress(d: *Display, event: hf.ProgressEvent, now: std.Io.Timestamp) ?Progress {
        const phase_changed = d.last_phase == null or d.last_phase.? != event.phase;
        const name: []const u8 = switch (event.phase) {
            .resolving => "resolving",
            .selection_required => "interrupted",
            .verifying => if (event.reused) "reused" else if (event.file_completed == 0) "checking" else "verifying",
            .downloading => if (phase_changed and event.resumed_bytes > 0) "resuming" else "downloading",
            .complete => "complete",
        };
        const done: u64 = switch (event.phase) {
            .resolving, .selection_required => 0,
            .verifying => if (event.reused or event.file_completed > 0) d.file_size else 0,
            .downloading => event.file_completed,
            .complete => d.file_size,
        };
        if (event.phase == .downloading and !phase_changed and event.file_completed < event.file_total) {
            if (d.last_line) |last| if (last.durationTo(now).toMilliseconds() < 250) return null;
        }
        if (event.phase == .downloading) d.last_line = now;
        return .{ .event = name, .file = d.file, .index = d.index, .count = d.count, .done = done, .size = d.file_size, .pull_done = d.done + done, .pull_size = d.total };
    }

    fn render(d: *Display, event: hf.ProgressEvent) !void {
        var buffer: [512]u8 = undefined;
        // Streaming, not positional: a positional writer restarts at offset
        // zero on every call and overwrites a redirected log.
        var stderr = std.Io.File.stderr().writerStreaming(d.io, &buffer);
        const w = &stderr.interface;
        defer w.flush() catch {};
        const now: std.Io.Timestamp = .now(d.io, .awake);
        if (d.json) {
            const line = d.progress(event, now);
            d.last_phase = event.phase;
            if (line) |l| {
                try std.json.Stringify.value(l, .{}, w);
                try w.writeByte('\n');
            }
            return;
        }
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
                try w.print("{s}: verifying the digest and publishing\n", .{name});
            },
            .downloading => {
                if (phase_changed or event.file_completed == 0) {
                    d.download_started = now;
                    d.download_base = event.file_completed;
                    if (event.file_count > 1) try w.print("file {d} of {d}\n", .{ event.file_index + 1, event.file_count });
                    if (event.resumed_bytes > 0) try w.print("{s}: resuming a kept partial download at {d:.1} of {d:.1} MB\n", .{ name, mb(event.resumed_bytes), mb(event.file_total) });
                    if (!d.tty) try w.print("{s}: downloading {d:.1} MB\n", .{ name, mb(event.file_total - event.resumed_bytes) });
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

/// A model file found under `<root>/models` in the layout.
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

/// One file of an installed model, as its sidecar records it.
pub const InstalledFile = struct {
    role: Role,
    variant: ?[]const u8 = null,
    /// Relative to the models directory.
    path: []const u8,
    size: u64,
    /// Against the index's pin for an index file, else `present` with a
    /// sidecar and `unverified` without one.
    status: catalog.Status,
    revision: ?[]const u8 = null,
    sha256: ?[]const u8 = null,
};

/// The options an index model runs with when nothing overrides them, as
/// `model ls -l` shows them.
pub const RunDefaults = struct {
    profile: ?config.Profile,
    speculative: bool,
    draft: ?[]const u8,
    draft_length: usize,
    think: ?inference.profiles.Effort,
    thinking_budget: ?usize,
};

/// A model on disk: an index artifact whose main file is present, or a
/// main file the index does not know, with the companions beside it.
pub const Installed = struct {
    /// The index name; for another file its registry name, else its
    /// repository path (`owner/repo`, or the file's path when a repository
    /// holds several).
    name: []const u8,
    /// Null when nothing tells what runs it (a safetensors directory of no
    /// known layout, a GGUF whose header does not parse).
    kind: ?catalog.ModelKind,
    /// The index artifact, null for a file not in the index.
    artifact: ?[]const u8 = null,
    legacy: bool = false,
    registered: ?RegisteredAs = null,
    architecture: ?[]const u8 = null,
    quantization: []const u8,
    /// Whether an adapter runs it; an index model always does.
    runnable: bool,
    /// The files present, summed.
    size: u64,
    /// `engine.model` names it.
    default: bool = false,
    files: []const InstalledFile,
    defaults: ?RunDefaults = null,
    warnings: []const []const u8 = &.{},
};

/// An index artifact and whether it is installed, for `model ls --available`.
pub const AvailableFile = struct {
    role: Role,
    variant: ?[]const u8,
    /// Where it lands, relative to the models directory.
    local: []const u8,
    repo: []const u8,
    revision: []const u8,
    remote: []const u8,
    size: u64,
    sha256: []const u8,
    used_for: []const u8,
    status: catalog.Status,
};

pub const Available = struct {
    name: []const u8,
    title: []const u8,
    kind: catalog.ModelKind,
    status: catalog.Lifecycle,
    repo: []const u8,
    revision: []const u8,
    architecture: []const u8,
    quantization: []const u8,
    /// The main file and those pulled with it.
    size: u64,
    installed: bool,
    files: []const AvailableFile,
    defaults: ?RunDefaults,
};

/// What `model ls` shows, before rendering.
pub const LsOptions = struct {
    long: bool = false,
    /// The index view instead of the installed one.
    available: bool = false,
    /// With `available`: legacy artifacts too.
    all: bool = false,
};

pub const Listing = struct {
    schema_version: u32 = 7,
    models_dir: []const u8,
    /// Every installed model's files, summed.
    total_bytes: u64,
    /// By kind (chat, speech, transcription, decision, embedding, other),
    /// then by name.
    installed: []const Installed,
    /// Model files with no sidecar (`nuclis model adopt <path>` writes one).
    without_sidecar: []const []const u8,
    /// Current index artifacts not installed.
    available: usize,
    /// Model files found above the `<owner>/<repo>/` level; they are not
    /// listed because nothing can say which repository they came from.
    outside_layout: usize,
    /// Registry entries whose file is absent.
    missing: []const MissingEntry = &.{},
    /// The index view, filled for `--available`.
    index: []const Available = &.{},

    pub fn find(self: Listing, name: []const u8) ?*const Installed {
        for (self.installed) |*i| if (std.mem.eql(u8, i.name, name) or (i.artifact != null and std.mem.eql(u8, i.artifact.?, name)) or (i.registered != null and std.mem.eql(u8, i.registered.?.name, name))) return i;
        return null;
    }

    pub fn render(self: Listing, out: *std.Io.Writer, options: LsOptions, json: bool, sty: style.Style) !void {
        if (json) {
            try std.json.Stringify.value(self, .{ .whitespace = .indent_2 }, out);
            return out.writeByte('\n');
        }
        if (options.available) return self.renderIndex(out, options, sty);
        const off = sty.off();
        var total: [16]u8 = undefined;
        try out.print("{s}models{s}  {s}{s}{s}  {s}{s}{s}\n", .{ sty.on(.label), off, sty.on(.code), self.models_dir, off, sty.on(.number), humanSize(&total, self.total_bytes), off });
        if (self.installed.len == 0) try out.print("\n{s}nothing installed{s} {s}(`nuclis model ls --available` lists the index; `nuclis model pull <name>` fetches one){s}\n", .{ sty.on(.bold), off, sty.on(.dim), off });
        const name_w = nameWidth(self.installed);
        var quant_w: usize = 0;
        for (self.installed) |i| quant_w = @max(quant_w, i.quantization.len);
        var group: ?[]const u8 = null;
        for (self.installed) |i| {
            const heading = if (i.kind) |k| catalog.groupName(k) else "other";
            if (group == null or !std.mem.eql(u8, group.?, heading)) {
                try out.print("\n{s}{s}{s}\n", .{ sty.on(.header), heading, off });
                group = heading;
            }
            try out.writeAll("  ");
            var short: [max_name_cells]u8 = undefined;
            try column(out, sty.on(.keyword), shorten(&short, i.name), off, name_w);
            try column(out, sty.on(.dim), i.quantization, off, quant_w);
            var size: [16]u8 = undefined;
            try out.print("{s}{s:>9}{s}", .{ sty.on(.number), humanSize(&size, i.size), off });
            var companions: [128]u8 = undefined;
            const tags = companionTags(&companions, i.files);
            if (tags.len > 0) try out.print("  {s}{s}{s}", .{ sty.on(.dim), tags, off });
            if (i.artifact == null) try out.print("  {s}not in the index{s}", .{ sty.on(.dim), off });
            if (i.artifact == null and !i.runnable) try out.print("  {s}not runnable{s}", .{ sty.on(.warning), off });
            if (i.legacy) try out.print("  {s}legacy{s}", .{ sty.on(.dim), off });
            if (i.registered) |r| if (!std.mem.eql(u8, r.name, i.name)) try out.print("  {s}registered as{s} {s}{s}{s}", .{ sty.on(.dim), off, sty.on(.keyword), r.name, off });
            if (i.default) try out.print("  {s}default{s}", .{ sty.on(.success), off });
            try out.writeByte('\n');
            for (i.warnings) |w| try out.print("    {s}{s}{s}\n", .{ sty.on(.warning), w, off });
            if (!options.long) continue;
            for (i.files) |f| {
                try out.writeAll("    ");
                var role: [48]u8 = undefined;
                const label = if (f.variant) |v| std.fmt.bufPrint(&role, "{s}:{s}", .{ @tagName(f.role), v }) catch @tagName(f.role) else @tagName(f.role);
                try column(out, sty.on(.number), label, off, 14);
                try out.print("{s}commit {s}{s}  {s}sha256 {s}{s}  {s}{s}{s}", .{ sty.on(.comment), if (f.revision) |r| r[0..12] else "-           ", off, sty.on(.comment), if (f.sha256) |h| h[0..12] else "-           ", off, sty.on(.code), f.path, off });
                if (f.status != .present) try out.print("  {s}{s}{s}", .{ sty.on(statusStyle(f.status)), @tagName(f.status), off });
                try out.writeByte('\n');
            }
            if (i.defaults) |d| try renderDefaults(out, d, sty);
        }
        if (self.missing.len > 0) {
            try out.print("\n{s}registry entries without a file{s} {s}(`nuclis config sync --prune` removes them){s}\n", .{ sty.on(.header), off, sty.on(.dim), off });
            var width: usize = 0;
            for (self.missing) |m| width = @max(width, m.name.len);
            for (self.missing) |m| {
                try out.writeAll("  ");
                try column(out, sty.on(.keyword), m.name, off, width);
                try out.print("{s}{s}{s}\n", .{ sty.on(.code), m.path, off });
            }
        }
        try out.writeByte('\n');
        if (self.without_sidecar.len > 0) try out.print("{s}{d} file{s} without a sidecar:{s} {s}nuclis model adopt <path>{s}\n", .{ sty.on(.warning), self.without_sidecar.len, if (self.without_sidecar.len == 1) "" else "s", off, sty.on(.code), off });
        if (self.available > 0) try out.print("{s}{d} more in the index:{s} {s}nuclis model ls --available{s}\n", .{ sty.on(.dim), self.available, off, sty.on(.code), off });
        if (self.outside_layout > 0) try out.print("{s}{d} model file(s) outside the <owner>/<repo>/ layout are not listed{s}\n", .{ sty.on(.warning), self.outside_layout, off });
    }

    fn renderIndex(self: Listing, out: *std.Io.Writer, options: LsOptions, sty: style.Style) !void {
        const off = sty.off();
        try out.print("{s}index{s} {s}(what nuclis pulls and verifies by name; `nuclis model pull <name> [--all]`){s}\n", .{ sty.on(.label), off, sty.on(.dim), off });
        var name_w: usize = 0;
        var quant_w: usize = 0;
        for (self.index) |a| {
            name_w = @max(name_w, a.name.len);
            quant_w = @max(quant_w, a.quantization.len);
        }
        var group: ?catalog.ModelKind = null;
        for (self.index) |a| {
            if (group != a.kind) {
                try out.print("\n{s}{s}{s}\n", .{ sty.on(.header), catalog.groupName(a.kind), off });
                group = a.kind;
            }
            try out.writeAll("  ");
            try column(out, sty.on(.keyword), a.name, off, name_w);
            try column(out, sty.on(.dim), a.quantization, off, quant_w);
            var size: [16]u8 = undefined;
            try out.print("{s}{s:>9}{s}  ", .{ sty.on(.number), humanSize(&size, a.size), off });
            if (a.installed) try out.print("{s}installed{s}", .{ sty.on(.success), off }) else try out.print("{s}not pulled{s}", .{ sty.on(.dim), off });
            if (a.status == .legacy) try out.print("  {s}legacy{s}", .{ sty.on(.dim), off });
            try out.writeByte('\n');
            if (!options.long) continue;
            for (a.files) |f| {
                try out.writeAll("    ");
                var role: [48]u8 = undefined;
                const label = if (f.variant) |v| std.fmt.bufPrint(&role, "{s}:{s}", .{ @tagName(f.role), v }) catch @tagName(f.role) else @tagName(f.role);
                try column(out, sty.on(.number), label, off, 14);
                var fsize: [16]u8 = undefined;
                try out.print("{s}{s:>9}{s}  {s}commit {s}  sha256 {s}{s}  {s}{s}{s}", .{ sty.on(.number), humanSize(&fsize, f.size), off, sty.on(.comment), f.revision[0..12], f.sha256[0..12], off, sty.on(.code), f.local, off });
                if (f.used_for.len > 0) try out.print("  {s}for {s}{s}", .{ sty.on(.dim), f.used_for, off });
                try out.writeByte('\n');
            }
            if (a.defaults) |d| try renderDefaults(out, d, sty);
        }
        if (!options.all) {
            var legacy: usize = 0;
            for (&catalog.artifacts) |a| legacy += @intFromBool(a.status == .legacy);
            if (legacy > 0) try out.print("\n{s}{d} legacy artifact(s) hidden:{s} {s}nuclis model ls --available --all{s}\n", .{ sty.on(.dim), legacy, off, sty.on(.code), off });
        }
    }
};

/// The longest name `model ls` prints whole; a longer one keeps its two
/// ends around an ellipsis.
const max_name_cells = 40;

fn nameWidth(installed: []const Installed) usize {
    var width: usize = 0;
    for (installed) |i| width = @max(width, @min(i.name.len, max_name_cells));
    return width;
}

/// `text` shortened in the middle to `max_name_cells` bytes.
fn shorten(buffer: *[max_name_cells]u8, text: []const u8) []const u8 {
    if (text.len <= max_name_cells) return text;
    const head = (max_name_cells - 3) / 2;
    const tail = max_name_cells - 3 - head;
    @memcpy(buffer[0..head], text[0..head]);
    @memcpy(buffer[head..][0..3], "...");
    @memcpy(buffer[head + 3 ..][0..tail], text[text.len - tail ..]);
    return buffer;
}

/// `+mmproj +draft:mtp,dflash2 +codec` from the files besides the main one.
fn companionTags(buffer: []u8, files: []const InstalledFile) []const u8 {
    var w: std.Io.Writer = .fixed(buffer);
    const order = [_]Role{ .mmproj, .draft, .codec, .backbone, .imatrix };
    for (order) |role| {
        var first = true;
        for (files) |f| if (f.role == role) {
            if (first) {
                if (w.end > 0) w.writeByte(' ') catch break;
                w.print("+{s}", .{@tagName(role)}) catch break;
            }
            if (f.variant) |v| w.print("{s}{s}", .{ if (first) ":" else ",", v }) catch break;
            first = false;
        };
    }
    return w.buffered();
}

fn renderDefaults(out: *std.Io.Writer, d: RunDefaults, sty: style.Style) !void {
    try out.print("    {s}runs with: speculative {s}", .{ sty.on(.dim), if (d.speculative) "on" else "off" });
    if (d.draft) |v| try out.print(" (draft {s}, length {d})", .{ v, d.draft_length }) else if (d.speculative) try out.print(" (length {d})", .{d.draft_length});
    if (d.think) |t| try out.print(", think {s}", .{@tagName(t)});
    if (d.thinking_budget) |b| try out.print(", thinking budget {d}", .{b});
    if (d.profile) |p| try out.print(", profile {s}", .{@tagName(p)});
    try out.print("{s}\n", .{sty.off()});
}

/// The defaults an index artifact runs with.
pub fn runDefaults(a: *const catalog.Artifact) ?RunDefaults {
    if (a.kind != .generation) return null;
    const v = a.verdict();
    return .{ .profile = a.defaults.profile, .speculative = v.speculative, .draft = if (a.draftCount() > 1) a.defaults.draft else null, .draft_length = v.draft_length, .think = a.defaults.think, .thinking_budget = a.defaults.thinking_budget };
}

/// A size for reading: decimal units, two decimals from a gigabyte up,
/// one below (`16.46 GB`, `931.1 MB`, `12.0 KB`, `512 B`). The JSON form
/// and the pull report keep exact bytes.
pub fn humanSize(buffer: []u8, bytes: u64) []const u8 {
    const b: f64 = @floatFromInt(bytes);
    return if (bytes >= 1_000_000_000)
        std.fmt.bufPrint(buffer, "{d:.2} GB", .{b / 1e9}) catch "?"
    else if (bytes >= 1_000_000)
        std.fmt.bufPrint(buffer, "{d:.1} MB", .{b / 1e6}) catch "?"
    else if (bytes >= 1_000)
        std.fmt.bufPrint(buffer, "{d:.1} KB", .{b / 1e3}) catch "?"
    else
        std.fmt.bufPrint(buffer, "{d} B", .{bytes}) catch "?";
}

/// One grid cell: styled text padded to `width` (plus one separating
/// space); a width of 0 pads nothing, for the last cell of a row.
fn column(out: *std.Io.Writer, on: []const u8, text: []const u8, off: []const u8, width: usize) !void {
    try out.print("{s}{s}{s}", .{ on, text, off });
    if (width > 0) try out.splatByteAll(' ', (width -| text.len) + 1);
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

/// A companion by its sidecar's role, else by the names the Hub uses
/// (`mmproj-*.gguf`, `mtp-*.gguf`, `dflash*.gguf`, `*imatrix*`); null for
/// a main file.
pub fn companionRole(f: Listed) ?Role {
    if (f.sidecar) |s| if (s.role != .main and s.role != .backbone) return s.role;
    return companionRoleOfName(std.fs.path.basename(f.path));
}

pub fn companionRoleOfName(name: []const u8) ?Role {
    var lower: [256]u8 = undefined;
    const n = @min(name.len, lower.len);
    const l = std.ascii.lowerString(lower[0..n], name[0..n]);
    if (std.mem.indexOf(u8, l, "mmproj") != null) return .mmproj;
    if (std.mem.startsWith(u8, l, "mtp") or std.mem.indexOf(u8, l, "-mtp") != null or std.mem.indexOf(u8, l, "dflash") != null) return .draft;
    if (std.mem.indexOf(u8, l, "imatrix") != null) return .imatrix;
    return null;
}

/// Whether `file` sits in `dir` or one level below it (`MTP/` in the
/// Hub's layouts).
fn besideOrBelow(dir: []const u8, file: []const u8) bool {
    const d = std.fs.path.dirname(file) orelse "";
    if (std.mem.eql(u8, d, dir)) return true;
    const parent = std.fs.path.dirname(d) orelse "";
    return std.mem.eql(u8, parent, dir);
}

fn kindOrder(kind: ?catalog.ModelKind) u8 {
    const k = kind orelse return 5;
    return switch (k) {
        .generation => 0,
        .speech => 1,
        .transcription => 2,
        .decision => 3,
        .embedding => 4,
    };
}

/// What a file outside the index is, from its header (GGUF) or the files
/// beside it (safetensors): its kind, architecture, and whether an adapter
/// runs it. Only the GGUF directory is read.
fn judgeLocal(arena: Allocator, io: std.Io, models: []const u8, path: []const u8) !struct { kind: ?catalog.ModelKind, architecture: ?[]const u8, runnable: bool } {
    const absolute = try std.fs.path.join(arena, &.{ models, path });
    if (hf.format(path) == .safetensors) {
        const dir = std.fs.path.dirname(absolute) orelse ".";
        if (try exists(io, try std.fs.path.join(arena, &.{ dir, "rl_agent_config.json" }))) return .{ .kind = .decision, .architecture = null, .runnable = true };
        if (try exists(io, try std.fs.path.join(arena, &.{ dir, "speech_tokenizer", "model.safetensors" }))) return .{ .kind = .speech, .architecture = null, .runnable = true };
        return .{ .kind = null, .architecture = null, .runnable = false };
    }
    var doc = inference.gguf.open(arena, io, absolute, .{}) catch return .{ .kind = null, .architecture = null, .runnable = false };
    defer doc.deinit();
    const arch = try arena.dupe(u8, doc.string("general.architecture") orelse "");
    if (std.mem.eql(u8, arch, inference.embed.architecture)) return .{ .kind = .embedding, .architecture = arch, .runnable = true };
    if (std.mem.eql(u8, arch, "drama3")) return .{ .kind = .speech, .architecture = arch, .runnable = true };
    const adapter = inference.models.adapterFor(arch);
    return .{ .kind = if (adapter != null) .generation else null, .architecture = if (arch.len > 0) arch else null, .runnable = adapter != null };
}

/// The quantization a file name ends in (`Some-Model-Q4_K_M.gguf` →
/// `Q4_K_M`), for a file the index does not describe.
fn quantizationOf(path: []const u8) []const u8 {
    if (hf.format(path) == .safetensors) return "safetensors";
    const stem = std.fs.path.stem(std.fs.path.basename(path));
    return if (std.mem.lastIndexOfScalar(u8, stem, '-')) |i| stem[i + 1 ..] else stem;
}

/// The installed models from a walk of `<root>/models` in the layout only
/// (owner directories, repository directories, then files, with
/// subdirectories such as `MTP/`): index artifacts first by their pinned
/// paths, then every other main file with the companions beside it.
/// Nothing is created; a missing directory lists nothing. `default_model`
/// marks the row `engine.model` names.
pub fn list(arena: Allocator, io: std.Io, root: []const u8, registry: config.Models, default_model: ?[]const u8) !Listing {
    const models = try modelsDir(arena, root);
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
    std.mem.sort(Listed, files.items, {}, struct {
        fn less(_: void, a: Listed, b: Listed) bool {
            return std.mem.lessThan(u8, a.path, b.path);
        }
    }.less);
    const claimed = try arena.alloc(bool, files.items.len);
    @memset(claimed, false);
    var installed: std.ArrayList(Installed) = .empty;
    var index: std.ArrayList(Available) = .empty;
    var available: usize = 0;
    for (&catalog.artifacts) |*a| {
        var present: std.ArrayList(InstalledFile) = .empty;
        var avail_files: std.ArrayList(AvailableFile) = .empty;
        var warnings: std.ArrayList([]const u8) = .empty;
        var main_present = false;
        var size: u64 = 0;
        for (a.files) |*f| {
            const local = try std.fs.path.join(arena, &.{ a.directoryOf(f), f.path });
            var status: catalog.Status = .absent;
            for (files.items, claimed) |listed, *taken| if (std.mem.eql(u8, listed.path, local)) {
                taken.* = true;
                const recorded = listed.sidecar;
                status = if (recorded) |r| if (std.mem.eql(u8, r.sha256, f.sha256) and r.size == f.size) .present else .mismatch else .unverified;
                if (recorded) |r| if (r.size != listed.size) try warnings.append(arena, try arena.print("{s}: the size differs from its sidecar's", .{local}));
                if (status == .mismatch) try warnings.append(arena, try arena.print("{s}: its sidecar records other content than the index pins (`nuclis model pull {s} --force` replaces it)", .{ local, a.name }));
                if (status == .unverified) try warnings.append(arena, try arena.print("{s}: no sidecar (`nuclis model adopt {s}` verifies it)", .{ local, local }));
                if (f.role == .main) main_present = true;
                size += listed.size;
                try present.append(arena, .{ .role = f.role, .variant = f.variant, .path = local, .size = listed.size, .status = status, .revision = if (recorded) |r| r.revision else null, .sha256 = if (recorded) |r| r.sha256 else null });
                break;
            };
            try avail_files.append(arena, .{ .role = f.role, .variant = f.variant, .local = local, .repo = a.origin(f), .revision = a.originRevision(f), .remote = catalog.Artifact.hubName(f), .size = f.size, .sha256 = f.sha256, .used_for = f.used_for, .status = status });
        }
        if (!main_present and present.items.len > 0) try warnings.append(arena, try arena.print("only companions are present; `nuclis model pull {s}` fetches the model", .{a.name}));
        if (main_present or present.items.len > 0) {
            try installed.append(arena, .{ .name = a.name, .kind = a.kind, .artifact = a.name, .legacy = a.status == .legacy, .architecture = a.architecture, .quantization = a.quantization, .runnable = true, .size = size, .files = present.items, .defaults = runDefaults(a), .warnings = warnings.items });
        } else if (a.status == .current) available += 1;
        try index.append(arena, .{ .name = a.name, .title = a.title, .kind = a.kind, .status = a.status, .repo = a.repo, .revision = a.revision, .architecture = a.architecture, .quantization = a.quantization, .size = a.totalSize(), .installed = main_present, .files = avail_files.items, .defaults = runDefaults(a) });
    }
    // Files outside the index: each main file a row, its companions with it.
    var without: std.ArrayList([]const u8) = .empty;
    for (files.items) |f| if (f.sidecar == null) try without.append(arena, f.path);
    for (files.items, claimed) |f, *taken| {
        if (taken.* or companionRole(f) != null) continue;
        taken.* = true;
        const dir = std.fs.path.dirname(f.path) orelse "";
        var row_files: std.ArrayList(InstalledFile) = .empty;
        try row_files.append(arena, localFile(f, .main));
        var size: u64 = f.size;
        for (files.items, claimed) |c, *c_taken| {
            if (c_taken.* or !besideOrBelow(dir, c.path)) continue;
            const role = companionRole(c) orelse continue;
            c_taken.* = true;
            size += c.size;
            try row_files.append(arena, localFile(c, role));
        }
        const judged = try judgeLocal(arena, io, models, f.path);
        var warnings: std.ArrayList([]const u8) = .empty;
        if (f.sidecar) |s| if (s.size != f.size) try warnings.append(arena, try arena.print("{s}: the size differs from its sidecar's", .{f.path}));
        try installed.append(arena, .{ .name = f.path, .kind = judged.kind, .architecture = judged.architecture, .quantization = quantizationOf(f.path), .runnable = judged.runnable, .size = size, .files = row_files.items, .warnings = warnings.items });
    }
    // A companion with no main file beside it is a row of its own.
    for (files.items, claimed) |f, taken| if (!taken) {
        try installed.append(arena, .{ .name = f.path, .kind = null, .quantization = quantizationOf(f.path), .runnable = false, .size = f.size, .files = try arena.dupe(InstalledFile, &.{localFile(f, companionRole(f) orelse .main)}), .warnings = try arena.dupe([]const u8, &.{"a companion without its model beside it"}) });
    };
    // Rows outside the index are named by repository when it holds one main file.
    for (installed.items) |*row| if (row.artifact == null and row.files[0].role == .main) {
        const repo = repoOf(row.name) orelse continue;
        var mains: usize = 0;
        for (installed.items) |other| if (other.artifact == null and repoOf(other.name) != null and std.mem.eql(u8, repoOf(other.name).?, repo)) {
            mains += 1;
        };
        if (mains == 1) row.name = repo;
    };
    // The registry's view: an entry locating a row's main file names it.
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
        const found = for (installed.items) |*row| {
            if (!std.mem.eql(u8, row.files[0].path, location) or row.files[0].role != .main) continue;
            row.registered = as;
            if (row.artifact == null) row.name = named.name;
            if (row.kind == null) row.kind = e.kind orelse .generation;
            break true;
        } else false;
        if (!found) try missing.append(arena, .{ .name = named.name, .path = location });
    }
    var total: u64 = 0;
    for (installed.items) |*row| {
        total += row.size;
        if (default_model) |d| row.default = std.mem.eql(u8, row.name, d) or (row.registered != null and std.mem.eql(u8, row.registered.?.name, d));
    }
    std.mem.sort(Installed, installed.items, {}, struct {
        fn less(_: void, a: Installed, b: Installed) bool {
            const ka = kindOrder(a.kind);
            const kb = kindOrder(b.kind);
            if (ka != kb) return ka < kb;
            return std.mem.lessThan(u8, a.name, b.name);
        }
    }.less);
    std.mem.sort(Available, index.items, {}, struct {
        fn less(_: void, a: Available, b: Available) bool {
            const ka = kindOrder(a.kind);
            const kb = kindOrder(b.kind);
            if (ka != kb) return ka < kb;
            return std.mem.lessThan(u8, a.name, b.name);
        }
    }.less);
    return .{ .models_dir = models, .total_bytes = total, .installed = installed.items, .without_sidecar = without.items, .available = available, .outside_layout = outside, .missing = missing.items, .index = index.items };
}

fn localFile(f: Listed, role: Role) InstalledFile {
    return .{ .role = role, .variant = if (f.sidecar) |s| s.variant else null, .path = f.path, .size = f.size, .status = if (f.sidecar != null) .present else .unverified, .revision = if (f.sidecar) |s| s.revision else null, .sha256 = if (f.sidecar) |s| s.sha256 else null };
}

/// `owner/repo` of a path in the layout.
fn repoOf(path: []const u8) ?[]const u8 {
    const first = std.mem.indexOfScalar(u8, path, '/') orelse return null;
    const second = std.mem.indexOfScalarPos(u8, path, first + 1, '/') orelse return null;
    return path[0..second];
}

/// A model the agent can switch to (`/model`): a registry entry or an
/// index name, as `--model` would name it.
pub const Runnable = struct {
    name: []const u8,
    /// Architecture, quantization, and size for an index file; the file's
    /// name otherwise.
    detail: []const u8,
};

/// The chat models whose files are present, as `listing` found them:
/// registry entries in the file's order, then index names no entry
/// shadows. Other kinds, absent files, and index files whose digest
/// disagrees with the index are left out.
pub fn runnableModels(arena: Allocator, listing: Listing, registry: config.Models) ![]const Runnable {
    var out: std.ArrayList(Runnable) = .empty;
    entries: for (registry.entries) |named| {
        if ((named.entry.kind orelse .generation) != .generation) continue;
        for (listing.missing) |m| if (std.mem.eql(u8, m.name, named.name)) continue :entries;
        const e = named.entry;
        const location = e.path orelse if (e.repo != null and e.file != null) try std.fs.path.join(arena, &.{ e.repo.?, e.file.? }) else continue;
        const detail = for (listing.installed) |row| {
            if (row.artifact != null and row.kind == .generation and std.mem.eql(u8, row.files[0].path, location)) break try installedDetail(arena, row);
        } else std.fs.path.basename(location);
        try out.append(arena, .{ .name = named.name, .detail = detail });
    }
    for (listing.installed) |row| {
        const name = row.artifact orelse continue;
        if (row.kind != .generation or registry.find(name) != null) continue;
        if (row.files[0].role != .main or row.files[0].status == .mismatch) continue;
        try out.append(arena, .{ .name = name, .detail = try installedDetail(arena, row) });
    }
    return out.items;
}

fn installedDetail(arena: Allocator, row: Installed) ![]const u8 {
    var size: [16]u8 = undefined;
    return arena.print("{s} · {s} · {s}", .{ row.architecture orelse "?", row.quantization, humanSize(&size, row.size) });
}

fn collect(arena: Allocator, io: std.Io, dir: std.Io.Dir, prefix: []const u8, depth: u8, files: *std.ArrayList(Listed), outside: *usize) !void {
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        const is_model = hf.format(entry.name) != null;
        switch (entry.kind) {
            .directory => {
                if (depth >= 2 + max_depth_below_repo) continue;
                var child = dir.openDir(io, entry.name, .{ .iterate = true }) catch continue;
                defer child.close(io);
                const child_prefix = try std.mem.concat(arena, u8, &.{ prefix, entry.name, "/" });
                try collect(arena, io, child, child_prefix, depth + 1, files, outside);
            },
            .file => {
                if (!is_model) continue;
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

pub fn ls(gpa: Allocator, io: std.Io, root: []const u8, registry: config.Models, default_model: ?[]const u8, options: LsOptions, json: bool, out: *std.Io.Writer, sty: style.Style) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    var listing = try list(arena_state.allocator(), io, root, registry, default_model);
    if (options.available) {
        if (!options.all) {
            var current: std.ArrayList(Available) = .empty;
            for (listing.index) |a| if (a.status == .current) try current.append(arena_state.allocator(), a);
            listing.index = current.items;
        }
        if (json) {
            try std.json.Stringify.value(.{ .schema_version = 1, .models_dir = listing.models_dir, .artifacts = listing.index }, .{ .whitespace = .indent_2 }, out);
            return out.writeByte('\n');
        }
    } else listing.index = &.{};
    try listing.render(out, options, json, sty);
}

// ---- adopt -----------------------------------------------------------------

pub const AdoptReport = struct {
    schema_version: u32 = 1,
    path: []const u8,
    sidecar: []const u8,
    size: u64,
    sha256: []const u8,
    /// The index artifact and file the digest matched; null for none.
    artifact: ?[]const u8,
    role: Role,
    variant: ?[]const u8,
    /// The pinned origin, when the digest matched.
    repo: []const u8,
    revision: ?[]const u8,

    pub fn render(self: AdoptReport, out: *std.Io.Writer, json: bool, sty: style.Style) !void {
        if (json) {
            try std.json.Stringify.value(self, .{ .whitespace = .indent_2 }, out);
            return out.writeByte('\n');
        }
        const off = sty.off();
        var size: [16]u8 = undefined;
        try out.print("{s}adopted{s} {s}{s}{s}  {s}{s}{s}  {s}sha256 {s}{s}\n", .{ sty.on(.success), off, sty.on(.code), self.path, off, sty.on(.number), humanSize(&size, self.size), off, sty.on(.comment), self.sha256[0..12], off });
        if (self.artifact) |a| {
            try out.print("  {s}the index's {s}{s}{s} file of {s}{s}{s} from {s} at {s}{s}\n", .{ sty.on(.dim), if (self.variant) |v| v else @tagName(self.role), off, sty.on(.keyword), a, off, sty.on(.dim), self.repo, self.revision.?[0..12], off });
        } else try out.print("  {s}matches no pinned file: the sidecar keeps its local digest so the file stays identity-stable{s}\n", .{ sty.on(.dim), off });
    }
};

/// `model adopt <path>`: hashes a file under `<root>/models`, finds the
/// index file with that size and SHA-256, and writes its sidecar (role,
/// variant, origin, `adopted_at`); with no match, a sidecar with the local
/// digest and no commit. A file that already has a valid sidecar is
/// refused unless `force`.
pub fn adopt(gpa: Allocator, io: std.Io, root: []const u8, path: []const u8, force: bool, json: bool, out: *std.Io.Writer, sty: style.Style, diag: *config.Diagnostic) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const models = try modelsDir(arena, root);
    const absolute = if (std.fs.path.isAbsolute(path)) try std.fs.path.resolve(arena, &.{path}) else try std.fs.path.resolve(arena, &.{ models, path });
    if (!inside(models, absolute)) {
        diag.set("{s} is not under the models directory {s}", .{ path, models });
        return error.OutsideModelsDirectory;
    }
    const relative = absolute[models.len + 1 ..];
    const repo = repoOf(relative) orelse {
        diag.set("{s} is outside the <owner>/<repo>/ layout; move it under a repository directory first", .{relative});
        return error.OutsideModelsDirectory;
    };
    if (hf.format(relative) == null) {
        diag.set("{s} is not a model file (.gguf or .safetensors)", .{relative});
        return error.NotAModelFile;
    }
    const sidecar_path = try sidecarPath(arena, absolute);
    if (!force) if (readSidecar(arena, io, .cwd(), sidecar_path) catch null) |_| {
        diag.set("{s} already has a sidecar; pass --force to write it again", .{relative});
        return error.AlreadyAdopted;
    };
    const stat = std.Io.Dir.cwd().statFile(io, absolute, .{}) catch |err| switch (err) {
        error.FileNotFound => {
            diag.set("{s}: no such file", .{relative});
            return error.FileNotFound;
        },
        else => return err,
    };
    const digest = try fileDigest(arena, io, absolute);
    var stamp: [20]u8 = undefined;
    const adopted_at = rfc3339(&stamp, std.Io.Timestamp.now(io, .real).toSeconds());
    const match = catalog.findDigest(digest, stat.size);
    // A file the index pins needs no header; another must at least parse.
    const header = if (match == null and hf.format(relative) == .gguf) roleFromHeader(gpa, io, absolute) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => {
            diag.set("{s}: not a GGUF file nuclis reads ({s}); nothing written", .{ relative, @errorName(err) });
            return error.NotAModelFile;
        },
    } else null;
    const role: Role = if (match) |m| m.file.role else header orelse companionRoleOfName(std.fs.path.basename(relative)) orelse .main;
    const sidecar: Sidecar = if (match) |m| .{
        .repo = m.artifact.origin(m.file),
        .file = catalog.Artifact.hubName(m.file),
        .revision = m.artifact.originRevision(m.file),
        .sha256 = digest,
        .size = stat.size,
        .role = role,
        .artifact = m.artifact.name,
        .variant = m.file.variant,
        .adopted_at = adopted_at,
        .nuclis_version = version,
    } else .{ .repo = repo, .file = relative[repo.len + 1 ..], .revision = null, .sha256 = digest, .size = stat.size, .role = role, .adopted_at = adopted_at, .nuclis_version = version };
    try writeSidecar(arena, io, .cwd(), sidecar_path, sidecar);
    try (AdoptReport{ .path = relative, .sidecar = sidecar_path, .size = stat.size, .sha256 = digest, .artifact = if (match) |m| m.artifact.name else null, .role = role, .variant = sidecar.variant, .repo = sidecar.repo, .revision = sidecar.revision }).render(out, json, sty);
}

/// The SHA-256 of a whole file as 64 lowercase hex digits.
pub fn fileDigest(arena: Allocator, io: std.Io, path: []const u8) ![]const u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var buffer: [256 * 1024]u8 = undefined;
    var reader = file.reader(io, &buffer);
    var hash: std.crypto.hash.sha2.Sha256 = .init(.{});
    var chunk: [256 * 1024]u8 = undefined;
    while (true) {
        const n = try reader.interface.readSliceShort(&chunk);
        if (n == 0) break;
        hash.update(chunk[0..n]);
    }
    return arena.dupe(u8, &std.fmt.bytesToHex(hash.finalResult(), .lower));
}

// ---- rm --------------------------------------------------------------------

/// A file `rm` deleted (a model file or a sidecar) and the bytes it held.
pub const Removed = struct { path: []const u8, bytes: u64 };
/// A file `rm` left in place because another present entry uses it.
pub const Kept = struct { path: []const u8, used_by: []const u8 };

pub const RmReport = struct {
    schema_version: u32 = 1,
    name: []const u8,
    models_dir: []const u8,
    removed: []const Removed,
    kept: []const Kept,
    freed_bytes: u64,

    pub fn render(self: RmReport, out: *std.Io.Writer, json: bool, sty: style.Style) !void {
        if (json) {
            try std.json.Stringify.value(self, .{ .whitespace = .indent_2 }, out);
            return out.writeByte('\n');
        }
        const off = sty.off();
        for (self.removed) |r| {
            var size: [16]u8 = undefined;
            try out.print("{s}removed{s} {s}{s}{s}  {s}{s}{s}\n", .{ sty.on(.success), off, sty.on(.code), self.relative(r.path), off, sty.on(.number), humanSize(&size, r.bytes), off });
        }
        for (self.kept) |k| try out.print("{s}kept{s}    {s}{s}{s}  {s}used by{s} {s}{s}{s}\n", .{ sty.on(.warning), off, sty.on(.code), self.relative(k.path), off, sty.on(.dim), off, sty.on(.keyword), k.used_by, off });
        var size: [16]u8 = undefined;
        try out.print("{s}{s}{s}: {s}{s}{s} freed\n", .{ sty.on(.keyword), self.name, off, sty.on(.number), humanSize(&size, self.freed_bytes), off });
    }

    fn relative(self: RmReport, path: []const u8) []const u8 {
        return if (inside(self.models_dir, path)) path[self.models_dir.len + 1 ..] else path;
    }
};

/// Which companions a removal takes, as `pull` takes them.
const Selection = struct {
    with: std.EnumSet(Role) = .empty,
    all: bool = false,

    const every: Selection = .{ .all = true };

    fn takes(self: Selection, role: Role) bool {
        return self.all or self.with.contains(role);
    }
};

/// `model rm <name>`: deletes a catalogue or registry entry's files and
/// their sidecars from the models directory, keeping any file another
/// present entry uses, then the directories the removal emptied. The
/// registry is not edited. A same-named registry entry is the catalogue
/// entry's twin (`config init` writes one per catalogue model), not a user.
pub fn rm(gpa: Allocator, io: std.Io, root: []const u8, registry: config.Models, options: PullOptions, json: bool, out: *std.Io.Writer, sty: style.Style, diag: *config.Diagnostic) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const report = try remove(arena_state.allocator(), io, root, registry, options.repo, .{ .with = options.with, .all = options.all }, diag);
    try report.render(out, json, sty);
}

fn remove(arena: Allocator, io: std.Io, root: []const u8, registry: config.Models, name: []const u8, selection: Selection, diag: *config.Diagnostic) !RmReport {
    if (name.len == 0) return error.MissingRepository;
    const models = try modelsDir(arena, root);
    const registered = registry.find(name);
    const targets = (try catalogueFiles(arena, io, models, name, selection, diag)) orelse blk: {
        const entry = registered orelse {
            if (catalog.renamedAudio(name)) |now| {
                diag.set("{s} is now {s}: a speech model's name says CustomVoice or Base", .{ name, now });
                return error.UnknownModel;
            }
            diag.set("{s} is not an index name or an entry of the models registry", .{name});
            return error.UnknownModel;
        };
        break :blk try registryFiles(arena, io, models, name, entry, selection, diag);
    };
    for (targets) |path| if (!inside(models, path)) {
        diag.set("{s}: {s} is outside the models directory; remove it by hand", .{ name, path });
        return error.OutsideModelsDirectory;
    };
    if (!try anyPresent(arena, io, targets)) {
        const whole = (try catalogueFiles(arena, io, models, name, .every, diag)) orelse
            try registryFiles(arena, io, models, name, registered.?, .every, diag);
        if (!selection.all and try anyPresent(arena, io, whole)) {
            diag.set("{s}: only its companions are present; --all removes them", .{name});
        } else diag.set("{s} has nothing pulled in {s}", .{ name, models });
        return error.NotPulled;
    }
    // Every other entry whose main file is present, with all its files.
    var users: std.ArrayList(Kept) = .empty;
    for (try catalogueNames(arena)) |other| {
        if (std.mem.eql(u8, other, name)) continue;
        const files = (try catalogueFiles(arena, io, models, other, .every, diag)).?;
        if (!try exists(io, files[0])) continue;
        for (files) |f| try users.append(arena, .{ .path = f, .used_by = other });
    }
    for (registry.entries) |*named| {
        if (std.mem.eql(u8, named.name, name)) continue;
        const files = try registryFiles(arena, io, models, named.name, &named.entry, .every, diag);
        if (files.len == 0 or !try exists(io, files[0])) continue;
        for (files) |f| try users.append(arena, .{ .path = f, .used_by = named.name });
    }
    var removed: std.ArrayList(Removed) = .empty;
    var kept: std.ArrayList(Kept) = .empty;
    var freed: u64 = 0;
    targets: for (targets) |path| {
        for (users.items) |u| if (std.mem.eql(u8, u.path, path)) {
            if (try exists(io, path)) try kept.append(arena, .{ .path = path, .used_by = u.used_by });
            continue :targets;
        };
        for ([_][]const u8{ path, try sidecarPath(arena, path) }) |file| {
            const bytes = try deleteCounted(io, file) orelse continue;
            try removed.append(arena, .{ .path = file, .bytes = bytes });
            freed += bytes;
        }
    }
    for (removed.items) |r| try tidy(io, models, std.fs.path.dirname(r.path).?);
    return .{ .name = name, .models_dir = models, .removed = removed.items, .kept = kept.items, .freed_bytes = freed };
}

/// Whether `path` lies strictly below `dir` (both normalized and absolute).
fn inside(dir: []const u8, path: []const u8) bool {
    return path.len > dir.len + 1 and std.mem.startsWith(u8, path, dir) and path[dir.len] == '/';
}

fn exists(io: std.Io, path: []const u8) !bool {
    std.Io.Dir.cwd().access(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    return true;
}

fn anyPresent(arena: Allocator, io: std.Io, files: []const []const u8) !bool {
    for (files) |f| if (try exists(io, f) or try exists(io, try sidecarPath(arena, f))) return true;
    return false;
}

/// The file's size, after deleting it; null when it was not there.
fn deleteCounted(io: std.Io, path: []const u8) !?u64 {
    const stat = std.Io.Dir.cwd().statFile(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    try std.Io.Dir.cwd().deleteFile(io, path);
    return stat.size;
}

/// Removes `start` and each parent below `models` while it is empty (a
/// Finder `.DS_Store` alone counts as empty).
fn tidy(io: std.Io, models: []const u8, start: []const u8) !void {
    var dir = start;
    while (inside(models, dir)) : (dir = std.fs.path.dirname(dir).?) {
        if (std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true })) |opened| {
            var d = opened;
            defer d.close(io);
            var it = d.iterate();
            while (try it.next(io)) |entry| if (!std.mem.eql(u8, entry.name, ".DS_Store")) return;
            d.deleteFile(io, ".DS_Store") catch |err| switch (err) {
                error.FileNotFound => {},
                else => return err,
            };
        } else |err| switch (err) {
            error.FileNotFound => continue,
            else => return err,
        }
        std.Io.Dir.cwd().deleteDir(io, dir) catch |err| switch (err) {
            error.FileNotFound => {},
            error.DirNotEmpty => return,
            else => return err,
        };
    }
}

fn catalogueNames(arena: Allocator) ![]const []const u8 {
    var names: std.ArrayList([]const u8) = .empty;
    for (&catalog.artifacts) |a| try names.append(arena, a.name);
    return names.items;
}

/// Appends `<models>/<parts...>` normalized (an absolute part wins), once.
fn addFile(arena: Allocator, files: *std.ArrayList([]const u8), models: []const u8, parts: []const []const u8) !void {
    const joined = try std.fs.path.join(arena, parts);
    const path = try std.fs.path.resolveAlloc(arena, &.{ models, joined });
    for (files.items) |f| if (std.mem.eql(u8, f, path)) return;
    try files.append(arena, path);
}

fn noSuchCompanion(name: []const u8, with: std.EnumSet(Role), has: std.EnumSet(Role), diag: *config.Diagnostic) !void {
    var wanted = with.iterator();
    while (wanted.next()) |role| if (!has.contains(role)) {
        diag.set("{s} has no {s} companion", .{ name, @tagName(role) });
        return error.NoSuchCompanion;
    };
}

/// An index artifact's files, main file first: its support files, codec,
/// and backbone always, its companions as `selection` takes them (every
/// draft variant when drafts are taken). Null for a name the index does
/// not hold.
fn catalogueFiles(arena: Allocator, io: std.Io, models: []const u8, name: []const u8, selection: Selection, diag: *config.Diagnostic) !?[]const []const u8 {
    const a = catalog.find(name) orelse return null;
    var has: std.EnumSet(Role) = .empty;
    for (a.files) |f| if (!catalog.pulledWith(f.role)) has.insert(f.role);
    try noSuchCompanion(name, selection.with, has, diag);
    var files: std.ArrayList([]const u8) = .empty;
    for (a.files) |*f| {
        if (!catalog.pulledWith(f.role) and !selection.takes(f.role)) continue;
        const first = files.items.len;
        try addFile(arena, &files, models, &.{ a.directoryOf(f), f.path });
        for (f.support) |support| try addFile(arena, &files, models, &.{ a.directoryOf(f), support });
        if (f.support.len > 0 or hf.format(f.path) == .safetensors) try discoverSupport(arena, io, models, files.items[first], a.origin(f), &files);
    }
    return files.items;
}

/// A registry entry's files, located as `model ls` locates them: the main
/// file, its support files when it is safetensors, and the `mmproj`/`mtp`
/// it names (relative to the main file's directory) as `selection` takes
/// them. Empty for an entry that locates no file.
fn registryFiles(arena: Allocator, io: std.Io, models: []const u8, name: []const u8, entry: *const config.ModelEntry, selection: Selection, diag: *config.Diagnostic) ![]const []const u8 {
    var has: std.EnumSet(Role) = .empty;
    if (entry.mmproj != null) has.insert(.mmproj);
    if (entry.draft != null) has.insert(.draft);
    try noSuchCompanion(name, selection.with, has, diag);
    var files: std.ArrayList([]const u8) = .empty;
    if (entry.path) |p| {
        try addFile(arena, &files, models, &.{p});
    } else if (entry.repo != null and entry.file != null) {
        try addFile(arena, &files, models, &.{ entry.repo.?, entry.file.? });
    } else return &.{};
    const main = files.items[0];
    if (hf.format(main) == .safetensors) {
        const recorded = readSidecar(arena, io, .cwd(), try sidecarPath(arena, main)) catch null;
        if (entry.repo orelse if (recorded) |r| r.repo else null) |repo| try discoverSupport(arena, io, models, main, repo, &files);
    }
    const dir = std.fs.path.dirname(main).?;
    for ([_]struct { role: Role, file: ?[]const u8 }{ .{ .role = .mmproj, .file = entry.mmproj }, .{ .role = .draft, .file = entry.draft } }) |c| {
        const file = c.file orelse continue;
        if (selection.takes(c.role)) try addFile(arena, &files, models, &.{ dir, file });
    }
    return files.items;
}

/// Support files pulled with safetensors weights: those beside the weights
/// and in subdirectories that are not another checkpoint's (the package's
/// `select` rule), found by a sidecar recording role `support` from `repo`.
fn discoverSupport(arena: Allocator, io: std.Io, models: []const u8, weights: []const u8, repo: []const u8, files: *std.ArrayList([]const u8)) !void {
    const path = std.fs.path.dirname(weights).?;
    var dir = std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound, error.NotDir => return,
        else => return err,
    };
    defer dir.close(io);
    // The index's checkpoint directories, which may nest in this one.
    var foreign: std.ArrayList([]const u8) = .empty;
    for (&catalog.artifacts) |*a| for (a.files) |*f| if (hf.format(f.path) == .safetensors) {
        try addFile(arena, &foreign, models, &.{ a.directoryOf(f), std.fs.path.dirnamePosix(f.path) orelse "" });
    };
    try walkSupport(arena, io, dir, path, 0, repo, foreign.items, files);
}

fn walkSupport(arena: Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8, depth: u8, repo: []const u8, foreign: []const []const u8, files: *std.ArrayList([]const u8)) !void {
    if (depth > 0) {
        for (foreign) |f| if (std.mem.eql(u8, f, path)) return;
        // Weights, or the sidecar left of them, mark another checkpoint.
        var probe = dir.iterate();
        while (try probe.next(io)) |entry| {
            const stem = if (std.mem.endsWith(u8, entry.name, sidecar_suffix)) entry.name[0 .. entry.name.len - sidecar_suffix.len] else entry.name;
            if (hf.format(stem) != null) return;
        }
    }
    var it = dir.iterate();
    while (try it.next(io)) |entry| switch (entry.kind) {
        .directory => {
            if (depth >= max_depth_below_repo) continue;
            var child = dir.openDir(io, entry.name, .{ .iterate = true }) catch continue;
            defer child.close(io);
            try walkSupport(arena, io, child, try std.fs.path.join(arena, &.{ path, entry.name }), depth + 1, repo, foreign, files);
        },
        .file => {
            if (!std.mem.endsWith(u8, entry.name, sidecar_suffix)) continue;
            const recorded = readSidecar(arena, io, dir, entry.name) catch |err| switch (err) {
                error.InvalidSidecar => continue,
                else => return err,
            } orelse continue;
            if (recorded.role != .support or !std.mem.eql(u8, recorded.repo, repo)) continue;
            const file = try std.fs.path.join(arena, &.{ path, entry.name[0 .. entry.name.len - sidecar_suffix.len] });
            for (files.items) |f| {
                if (std.mem.eql(u8, f, file)) break;
            } else try files.append(arena, file);
        },
        else => {},
    };
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
            diag.set("{s} is an index name and pins its file and commit; inspect by repository id ({s}) to choose another", .{ e.name, e.repo });
            return error.ConflictingOptions;
        }
        request = .{ .repo_id = e.repo, .revision = e.revision, .filename = e.main().path };
    }
    var remote = client.list(request) catch |err| return listFailure(err, arena, client, request, options.repo, diag);
    defer remote.deinit();
    // A safetensors checkpoint has no directory to read remotely: the
    // verdict is the index's digests against the Hub's.
    if (entry) |e| if (hf.format(e.main().path) == .safetensors) return renderInspect(out, try checkpointReport(arena, e, &remote), json, sty);
    const selected = (hf.select(arena, &remote, request.filename) catch |err| return hubFailure(err, options.repo, diag)) orelse {
        try renderSelection(out, .{ .repo = options.repo, .revision = &remote.revision, .selection_required = try choices(arena, remote.files) }, json, sty);
        try out.flush();
        diag.set("{s} has {d} model files; pass --file <name>", .{ options.repo, remote.files.len });
        return error.SelectionRequired;
    };
    // A split file's directory is in its first shard.
    const file = selected[0];
    if (hf.format(file.name) != .gguf) {
        diag.set("{s}: remote inspection reads GGUF directories; pull the safetensors artifact and inspect it locally", .{file.name});
        return error.UnsupportedFormat;
    }
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
        .sha256 = try arena.dupe(u8, &std.fmt.bytesToHex(file.sha256.?, .lower)),
    }, diag);
    try renderInspect(out, report, json, sty);
}

/// `inspect` of a safetensors checkpoint, from the Hub's listing at the
/// pinned commit: supported when the weights (and the files of its own
/// repository pulled with them) carry the index's digests.
fn checkpointReport(arena: Allocator, a: *const catalog.Artifact, remote: *const hf.Catalog) !InspectReport {
    const main_file = a.main();
    const weights = for (remote.files) |f| {
        if (std.mem.eql(u8, f.name, main_file.path)) break f;
    } else return error.FileNotFound;
    const digest = try hexOrNull(arena, weights.sha256) orelse "";
    var verdict: Verdict = .{ .status = .supported, .catalog = a.name, .role = .main, .reason = try arena.print("pinned by the index: a {s} checkpoint (safetensors, {s}) that `{s}` runs; `nuclis model pull {s}` fetches it with its support files{s}", .{ catalog.kindName(a.kind), a.quantization, catalog.command(a.kind), a.name, if (a.companion(.codec) != null) " and its codec" else "" }) };
    var mismatch: ?[]const u8 = if (!std.mem.eql(u8, digest, main_file.sha256)) main_file.path else null;
    for (a.files[1..]) |*c| if (c.repo == null) for (remote.files) |f| if (std.mem.eql(u8, f.name, c.path)) {
        if (!std.mem.eql(u8, try hexOrNull(arena, f.sha256) orelse "", c.sha256)) mismatch = c.path;
    };
    if (mismatch) |file| verdict = .{ .status = .not_runnable, .reason = try arena.print("{s}: the Hub's digest at the pinned commit is not the index's; the index entry needs updating", .{file}) };
    return finish(.{
        .name = a.name,
        .repo = a.repo,
        .file = main_file.path,
        .requested_revision = a.revision,
        .revision = try arena.dupe(u8, &remote.revision),
        .size = weights.size,
        .sha256 = digest,
    }, 0, 0, null, verdict);
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
                    try arena.print("tensor {s} uses encoding {s}, which nuclis does not store", .{ tensor, encoding })
                else
                    try arena.print("tensor {s} has a shape encoding {s} cannot store", .{ tensor, encoding });
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
    var snapshot = try inspection.snapshot(doc, try arena.print("{s}/{s}", .{ id.repo, id.file }), try arena.alloc(inspection.EncodingCount, 32));
    if (snapshot.architecture) |a| snapshot.architecture = try arena.dupe(u8, a);
    if (snapshot.name) |n| snapshot.name = try arena.dupe(u8, n);
    return finish(id, head.items.len, requests, snapshot, try judge(arena, gpa, &doc, id));
}

fn finish(id: Identity, bytes_read: u64, requests: usize, directory: ?inspection.Snapshot, verdict: Verdict) InspectReport {
    return .{ .name = id.name, .repo = id.repo, .file = id.file, .requested_revision = id.requested_revision, .revision = id.revision, .size = id.size, .sha256 = id.sha256, .bytes_read = bytes_read, .requests = requests, .directory = directory, .verdict = verdict };
}

/// "Q4_K (id 12)", or "id 999" for a layout nuclis does not know.
fn encodingLabel(arena: Allocator, encoding_id: u32) ![]const u8 {
    if (inference.encoding.layout(encoding_id)) |layout| return arena.print("{s} (id {d})", .{ layout.name, encoding_id });
    return arena.print("id {d}", .{encoding_id});
}

/// The verdict on a parsed directory. Runnable needs an adapter for the
/// architecture, every tensor in its executable set, and the adapter's
/// binding; supported adds the catalogue pinning the Hub's digest.
fn judge(arena: Allocator, gpa: Allocator, doc: *const inference.gguf.Document, id: Identity) !Verdict {
    const pinned = catalog.findFile(id.repo, id.file);
    const in_catalog = pinned != null and std.mem.eql(u8, pinned.?.file.sha256, id.sha256);
    const architecture = doc.string("general.architecture") orelse "";
    const adapter = inference.models.adapterFor(architecture) orelse {
        if (in_catalog and pinned.?.file.role != .main) return .{
            .status = .not_runnable,
            .catalog = pinned.?.artifact.name,
            .role = pinned.?.file.role,
            .reason = try arena.print("the {s} companion of index entry {s} (loaded for {s}, not by the text engine); architecture \"{s}\" has no adapter", .{ @tagName(pinned.?.file.role), pinned.?.artifact.name, pinned.?.file.used_for, architecture }),
        };
        return .{
            .status = .not_runnable,
            .reason = if (architecture.len == 0) "the file declares no general.architecture" else try arena.print("no adapter for architecture \"{s}\"", .{architecture}),
        };
    };
    for (doc.tensors) |tensor| if (!inference.models.registry.executableEncoding(adapter, tensor.encoding_id)) {
        const encoding = try encodingLabel(arena, tensor.encoding_id);
        return .{
            .status = .not_runnable,
            .reason = try arena.print("tensor {s} uses encoding {s}, outside the {s} adapter's executable set", .{ tensor.name, encoding, @tagName(adapter) }),
            .tensor = try arena.dupe(u8, tensor.name),
            .encoding = encoding,
        };
    };
    _ = inference.models.registry.validate(adapter, gpa, doc) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return .{ .status = .not_runnable, .reason = try arena.print("the {s} adapter rejects the file: {s}", .{ @tagName(adapter), @errorName(err) }) },
    };
    if (in_catalog and pinned.?.file.role == .main) return .{
        .status = .supported,
        .catalog = pinned.?.artifact.name,
        .role = .main,
        .reason = try arena.print("index entry {s}: the Hub's digest at this commit equals the pinned one, and the {s} adapter binds it", .{ pinned.?.artifact.name, @tagName(adapter) }),
    };
    return .{
        .status = .runnable,
        .reason = if (pinned != null)
            try arena.print("the {s} adapter binds it; the index pins another digest for this file (entry {s})", .{ @tagName(adapter), pinned.?.artifact.name })
        else
            try arena.print("the {s} adapter binds it; not in the index", .{@tagName(adapter)}),
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

test "--json progress: a line per phase, downloads at most every 250 ms and at the file's end" {
    const io = std.testing.io;
    const t0: std.Io.Timestamp = .now(io, .awake);
    var d: Display = .{ .io = io, .started = t0, .tty = false, .json = true, .file = "model.safetensors", .index = 1, .count = 3, .file_size = 100, .done = 40, .total = 200 };
    const resolving = d.progress(.{ .phase = .resolving }, t0).?;
    try std.testing.expectEqualStrings("resolving", resolving.event);
    d.last_phase = .resolving;
    const first = d.progress(.{ .phase = .downloading, .file_completed = 10, .file_total = 100 }, t0).?;
    try std.testing.expectEqual(@as(u64, 10), first.done);
    try std.testing.expectEqual(@as(u64, 50), first.pull_done);
    try std.testing.expectEqual(@as(u64, 200), first.pull_size);
    d.last_phase = .downloading;
    // Within 250 ms of the last line: nothing, unless the file is done.
    const soon = t0.addDuration(.fromMilliseconds(100));
    try std.testing.expect(d.progress(.{ .phase = .downloading, .file_completed = 20, .file_total = 100 }, soon) == null);
    try std.testing.expectEqual(@as(u64, 100), d.progress(.{ .phase = .downloading, .file_completed = 100, .file_total = 100 }, soon).?.done);
    const later = t0.addDuration(.fromMilliseconds(400));
    try std.testing.expect(d.progress(.{ .phase = .downloading, .file_completed = 60, .file_total = 100 }, later) != null);
    const reused = d.progress(.{ .phase = .verifying, .reused = true }, later).?;
    try std.testing.expectEqualStrings("reused", reused.event);
    try std.testing.expectEqual(@as(u64, 100), reused.done);
    // A file continued from a kept partial starts with its bytes as done.
    d.last_phase = .verifying;
    const resumed = d.progress(.{ .phase = .downloading, .file_completed = 30, .resumed_bytes = 30, .file_total = 100 }, later).?;
    try std.testing.expectEqualStrings("resuming", resumed.event);
    try std.testing.expectEqual(@as(u64, 30), resumed.done);
    d.last_phase = .downloading;
    const after = later.addDuration(.fromMilliseconds(400));
    try std.testing.expectEqualStrings("downloading", d.progress(.{ .phase = .downloading, .file_completed = 40, .resumed_bytes = 30, .file_total = 100 }, after).?.event);
    // One line of JSON each.
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try std.json.Stringify.value(reused, .{}, &out.writer);
    try std.testing.expectEqualStrings("{\"event\":\"reused\",\"file\":\"model.safetensors\",\"index\":1,\"count\":3,\"done\":100,\"size\":100,\"pull_done\":140,\"pull_size\":200}", out.written());
}

test "sizes read in decimal units, exact bytes below a kilobyte" {
    var buffer: [16]u8 = undefined;
    try std.testing.expectEqualStrings("16.46 GB", humanSize(&buffer, 16464440224));
    try std.testing.expectEqualStrings("931.1 MB", humanSize(&buffer, 931146432));
    try std.testing.expectEqualStrings("12.0 KB", humanSize(&buffer, 12000));
    try std.testing.expectEqualStrings("512 B", humanSize(&buffer, 512));
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
    try std.testing.expectEqual(Role.draft, try resolveRole(null, .draft));
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
        .role = .draft,
        .artifact = "qwen3.8-27b",
        .variant = "mtp",
        .downloaded_at = "2026-09-11T00:00:00Z",
        .nuclis_version = version,
    };
    try writeSidecar(arena, io, tmp.dir, name, written);
    const read = (try readSidecar(arena, io, tmp.dir, name)).?;
    try std.testing.expectEqualStrings(written.repo, read.repo);
    try std.testing.expectEqualStrings(written.revision.?, read.revision.?);
    try std.testing.expectEqualStrings(written.sha256, read.sha256);
    try std.testing.expectEqual(written.size, read.size);
    try std.testing.expectEqual(Role.draft, read.role);
    try std.testing.expectEqualStrings("mtp", read.variant.?);
    try std.testing.expectEqualStrings("qwen3.8-27b", read.artifact.?);
    try std.testing.expectEqualStrings(written.downloaded_at.?, read.downloaded_at.?);
    try std.testing.expect(read.git_blob == null);
    // A plain git file records its blob id; a sidecar without the field
    // (every one written before it existed) reads as null.
    var plain = written;
    plain.role = .support;
    plain.git_blob = "ef44a803696cdca06508fe711488cdc982eb64d3";
    try writeSidecar(arena, io, tmp.dir, "config.json.nuclis.json", plain);
    const plain_read = (try readSidecar(arena, io, tmp.dir, "config.json.nuclis.json")).?;
    try std.testing.expectEqualStrings(plain.git_blob.?, plain_read.git_blob.?);
    try std.testing.expectEqual(Role.support, plain_read.role);
    // Rewriting replaces the record (a second pull records its own commit).
    var again = written;
    again.revision = "89abcdef0123456789abcdef0123456789abcdef";
    try writeSidecar(arena, io, tmp.dir, name, again);
    try std.testing.expectEqualStrings(again.revision.?, (try readSidecar(arena, io, tmp.dir, name)).?.revision.?);
    // A schema 1 sidecar still reads, its `mtp` role as `draft`.
    try tmp.dir.writeFile(io, .{ .sub_path = name, .data = "{\"schema_version\":1,\"repo\":\"a/b\",\"file\":\"m\",\"revision\":\"0123456789abcdef0123456789abcdef01234567\",\"sha256\":\"322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482\",\"size\":1,\"role\":\"mtp\",\"downloaded_at\":\"2026-09-11T00:00:00Z\",\"nuclis_version\":\"0.1.0\"}" });
    const old = (try readSidecar(arena, io, tmp.dir, name)).?;
    try std.testing.expectEqual(Role.draft, old.role);
    try std.testing.expect(old.variant == null and old.adopted_at == null);
    // An adopted file nuclis could not match has no commit.
    var adopted = written;
    adopted.revision = null;
    adopted.downloaded_at = null;
    adopted.adopted_at = "2026-11-12T00:00:00Z";
    try writeSidecar(arena, io, tmp.dir, name, adopted);
    const adopted_read = (try readSidecar(arena, io, tmp.dir, name)).?;
    try std.testing.expect(adopted_read.revision == null and adopted_read.downloaded_at == null);
    try std.testing.expectEqualStrings("-", adopted_read.shortRevision());
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
    try std.testing.expect(std.mem.startsWith(u8, out.written(), "verified /m/a/b/Q4.gguf\n  4 bytes, sha256 ab, role main; provenance beside it in Q4.gguf.nuclis.json\n"));
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

test "ls lists what is installed, grouped by kind, with the index one flag away" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", alloc);
    defer alloc.free(root);
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // Absent directory: nothing installed, the whole index available, nothing created.
    const empty = try list(arena, io, root, .{}, null);
    try std.testing.expectEqual(@as(usize, 0), empty.installed.len);
    try std.testing.expectEqual(catalog.artifacts.len, empty.index.len);
    var current: usize = 0;
    for (&catalog.artifacts) |*a| current += @intFromBool(a.status == .current);
    try std.testing.expectEqual(current, empty.available);
    try std.testing.expectError(error.FileNotFound, tmp.dir.access(io, "models", .{}));
    const qwen = catalog.find("qwen3.8-27b").?;
    const q = "models/unsloth/Qwen3.8-27B-GGUF/";
    try tmp.dir.createDirPath(io, q ++ "MTP");
    try tmp.dir.createDirPath(io, "models/unsloth/Repo-GGUF/MTP");
    try tmp.dir.createDirPath(io, "models/other/Loose-GGUF");
    try tmp.dir.createDirPath(io, "models/qwen");
    // The pinned main file with a matching sidecar, the MTP head with a
    // wrong one, the DFlash2 drafter placed by hand (no sidecar).
    try tmp.dir.writeFile(io, .{ .sub_path = q ++ "Qwen3.8-27B-UD-Q4_K_M.gguf", .data = "GGUF" });
    try writeSidecar(arena, io, tmp.dir, q ++ "Qwen3.8-27B-UD-Q4_K_M.gguf.nuclis.json", .{ .repo = qwen.repo, .file = qwen.main().path, .revision = qwen.revision, .sha256 = qwen.main().sha256, .size = qwen.main().size, .role = .main, .artifact = qwen.name, .downloaded_at = "2026-09-11T00:00:00Z", .nuclis_version = version });
    try tmp.dir.writeFile(io, .{ .sub_path = q ++ "MTP/mtp-Qwen3.8-27B-Q4_0.gguf", .data = "GGUF" });
    try writeSidecar(arena, io, tmp.dir, q ++ "MTP/mtp-Qwen3.8-27B-Q4_0.gguf.nuclis.json", .{ .repo = qwen.repo, .file = "MTP/mtp-Qwen3.8-27B-Q4_0.gguf", .revision = qwen.revision, .sha256 = "0000000000000000000000000000000000000000000000000000000000000000", .size = 4, .role = .draft, .downloaded_at = "2026-09-11T00:00:00Z", .nuclis_version = version });
    try tmp.dir.writeFile(io, .{ .sub_path = q ++ "MTP/Qwen3.8-27B-DFlash2-Q8_0.gguf", .data = "GGUF" });
    // Outside the index: a main file with a sidecar and its drafter beside
    // it, a loose file, a safetensors set of no known layout, two above the layout.
    try tmp.dir.writeFile(io, .{ .sub_path = "models/unsloth/Repo-GGUF/Repo-Q4.gguf", .data = "GGUFxxxx" });
    try tmp.dir.writeFile(io, .{ .sub_path = "models/unsloth/Repo-GGUF/README.md", .data = "not listed" });
    try tmp.dir.writeFile(io, .{ .sub_path = "models/unsloth/Repo-GGUF/MTP/mtp-Repo.gguf", .data = "GGUF" });
    try tmp.dir.writeFile(io, .{ .sub_path = "models/other/Loose-GGUF/Loose-Q8_0.gguf", .data = "GGUF" });
    try tmp.dir.createDirPath(io, "models/acme/encoder/tokenizer");
    try tmp.dir.writeFile(io, .{ .sub_path = "models/acme/encoder/model.safetensors", .data = "\x02\x00\x00\x00\x00\x00\x00\x00{}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "models/acme/encoder/tokenizer/tokenizer.json", .data = "{}" });
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
    // forced profile, the index's own name, a second name for the index
    // file, one whose file is absent, and a path.
    const entries = [_]config.NamedModel{
        .{ .name = "repo", .entry = .{ .repo = "unsloth/Repo-GGUF", .file = "Repo-Q4.gguf", .profile = .gemma4 } },
        .{ .name = "qwen3.8-27b", .entry = .{ .repo = qwen.repo, .file = qwen.main().path } },
        .{ .name = "big", .entry = .{ .repo = qwen.repo, .file = qwen.main().path } },
        .{ .name = "gone", .entry = .{ .repo = "unsloth/Gone-GGUF", .file = "gone.gguf" } },
        .{ .name = "local", .entry = .{ .path = "other/Loose-GGUF/Loose-Q8_0.gguf" } },
    };
    const listing = try list(arena, io, root, .{ .entries = &entries }, "qwen3.8-27b");
    // Chat first, in name order (the registry names the files outside the
    // index), then what nothing says runs.
    try std.testing.expectEqual(@as(usize, 4), listing.installed.len);
    const names = [_][]const u8{ "local", "qwen3.8-27b", "repo", "acme/encoder" };
    for (names, listing.installed) |name, row| try std.testing.expectEqualStrings(name, row.name);
    const row = listing.find("qwen3.8-27b").?;
    try std.testing.expect(row.default and row.kind == .generation and row.artifact != null);
    try std.testing.expectEqualStrings("big", row.registered.?.name);
    // The present files only: the projector is not pulled.
    try std.testing.expectEqual(@as(usize, 3), row.files.len);
    try std.testing.expectEqual(catalog.Status.present, row.files[0].status);
    try std.testing.expectEqualStrings("mtp", row.files[1].variant.?);
    try std.testing.expectEqual(catalog.Status.mismatch, row.files[1].status);
    try std.testing.expectEqualStrings("unsloth/Qwen3.8-27B-GGUF/MTP/Qwen3.8-27B-DFlash2-Q8_0.gguf", row.files[2].path);
    try std.testing.expectEqual(catalog.Status.unverified, row.files[2].status);
    try std.testing.expectEqual(@as(u64, 12), row.size);
    const repo = listing.find("repo").?;
    try std.testing.expect(repo.artifact == null and repo.kind == .generation);
    try std.testing.expectEqual(.gemma4, repo.registered.?.profile.?);
    // A drafter beside a main file outside the index is its companion.
    try std.testing.expectEqual(@as(usize, 2), repo.files.len);
    try std.testing.expectEqual(Role.draft, repo.files[1].role);
    try std.testing.expectEqualStrings("0123456789abcdef0123456789abcdef01234567", repo.files[0].revision.?);
    const other = listing.find("acme/encoder").?;
    try std.testing.expect(other.kind == null and !other.runnable);
    try std.testing.expectEqual(@as(usize, 1), listing.missing.len);
    try std.testing.expectEqualStrings("gone", listing.missing[0].name);
    try std.testing.expectEqualStrings("unsloth/Gone-GGUF/gone.gguf", listing.missing[0].path);
    try std.testing.expectEqual(@as(usize, 2), listing.outside_layout);
    // Every model file without a sidecar, the index's hand-placed drafter included.
    try std.testing.expectEqual(@as(usize, 4), listing.without_sidecar.len);
    try std.testing.expectEqual(current - 1, listing.available);
    // What `/model` offers: the entries whose file is present, in the file's
    // order, then index names no entry shadows (none here: the one present
    // index model is registered under its own name too).
    const offered = try runnableModels(arena, listing, .{ .entries = &entries });
    try std.testing.expectEqual(@as(usize, 4), offered.len);
    try std.testing.expectEqualStrings("repo", offered[0].name);
    try std.testing.expectEqualStrings("Repo-Q4.gguf", offered[0].detail);
    try std.testing.expectEqualStrings("qwen3.8-27b", offered[1].name);
    try std.testing.expectEqualStrings("qwen35 · UD-Q4_K_M · 12 B", offered[1].detail);
    try std.testing.expectEqualStrings("big", offered[2].name);
    try std.testing.expectEqualStrings("local", offered[3].name);
    // Without the registry, the present index model is offered by name.
    const bare = try runnableModels(arena, try list(arena, io, root, .{}, null), .{});
    try std.testing.expectEqual(@as(usize, 1), bare.len);
    try std.testing.expectEqualStrings("qwen3.8-27b", bare[0].name);
    // An embedding entry is never offered to the text engine.
    const embedding_only = [_]config.NamedModel{.{ .name = "vectors", .entry = .{ .kind = .embedding, .path = "unsloth/Repo-GGUF/Repo-Q4.gguf" } }};
    const without = try runnableModels(arena, try list(arena, io, root, .{ .entries = &embedding_only }, null), .{ .entries = &embedding_only });
    try std.testing.expectEqual(@as(usize, 1), without.len);
    try std.testing.expectEqualStrings("qwen3.8-27b", without[0].name);

    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    try listing.render(&out.writer, .{}, false, .none);
    const text = out.written();
    try std.testing.expect(std.mem.indexOf(u8, text, "\nchat\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\nother\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "+draft:mtp,dflash2") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "registered as big  default\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "+draft  not in the index") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "not in the index  not runnable") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "its sidecar records other content than the index pins") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "registry entries without a file") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "  gone unsloth/Gone-GGUF/gone.gguf\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "4 files without a sidecar: nuclis model adopt <path>") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "more in the index: nuclis model ls --available") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "2 model file(s) outside") != null);
    // No digest without -l; with it, 12 characters of each and the defaults.
    try std.testing.expect(std.mem.indexOf(u8, text, "322e194ff797") == null);
    out.clearRetainingCapacity();
    try listing.render(&out.writer, .{ .long = true }, false, .none);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "commit 4ca720788d1e  sha256 322e194ff797  unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "draft:dflash2  commit -             sha256 -             unsloth/Qwen3.8-27B-GGUF/MTP/Qwen3.8-27B-DFlash2-Q8_0.gguf  unverified\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "runs with: speculative on (draft mtp, length 7), think low, thinking budget 1024, profile qwen38\n") != null);
    out.clearRetainingCapacity();
    try listing.render(&out.writer, .{}, true, .none);
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, out.written(), .{});
    defer parsed.deinit();
    try std.testing.expectEqual(@as(i64, 7), parsed.value.object.get("schema_version").?.integer);
    const rows = parsed.value.object.get("installed").?.array.items;
    try std.testing.expectEqualStrings("qwen3.8-27b", rows[1].object.get("name").?.string);
    // The JSON keeps whole digests.
    try std.testing.expectEqualStrings(qwen.main().sha256, rows[1].object.get("files").?.array.items[0].object.get("sha256").?.string);
    try std.testing.expectEqualStrings("gone", parsed.value.object.get("missing").?.array.items[0].object.get("name").?.string);

    // The index view: current artifacts by kind, installed ones marked, legacy behind --all.
    out.clearRetainingCapacity();
    try ls(alloc, io, root, .{}, null, .{ .available = true }, false, &out.writer, .none);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "qwen3.8-27b ") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "installed\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "gemma-4-12b-qat") == null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "legacy artifact(s) hidden") != null);
    out.clearRetainingCapacity();
    try ls(alloc, io, root, .{}, null, .{ .available = true, .all = true, .long = true }, false, &out.writer, .none);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "gemma-4-12b-qat") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "unsloth/Qwen3.8-27B-GGUF/MTP/Qwen3.8-27B-DFlash2-Q8_0.gguf") != null);
    out.clearRetainingCapacity();
    try ls(alloc, io, root, .{}, null, .{ .available = true }, true, &out.writer, .none);
    const index = try std.json.parseFromSlice(std.json.Value, alloc, out.written(), .{});
    defer index.deinit();
    try std.testing.expectEqual(current, index.value.object.get("artifacts").?.array.items.len);
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
        try w.writeInt(u32, @backingInt(m.kind), .little);
        switch (m.value) {
            .unsigned => |n| try w.writeInt(u32, @intCast(n), .little),
            .signed => |n| try w.writeInt(i32, @intCast(n), .little),
            .float => |f| try w.writeInt(u32, @bitCast(@as(f32, @floatCast(f))), .little),
            .boolean => |b| try w.writeInt(u8, @intFromBool(b), .little),
            .string => |text| try writeGgufString(w, if (mutation.architecture != null and std.mem.eql(u8, m.key, "general.architecture")) mutation.architecture.? else text),
            .array => |array| {
                try w.writeInt(u32, @backingInt(array.element_type), .little);
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
        .file = entry.main().path,
        .requested_revision = "main",
        .revision = entry.revision,
        // The weights would follow the aligned directory.
        .size = std.mem.alignForward(u64, directory.len, 32) + inventory.file_bytes,
        .sha256 = sha256,
    }, &diag);
}

test "--register knows a speech checkpoint from its config.json and refuses another" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", arena);
    const weights = try std.fs.path.join(arena, &.{ root, "model.safetensors" });
    const config_path = try std.fs.path.join(arena, &.{ root, "config.json" });
    const generation_path = try std.fs.path.join(arena, &.{ root, "generation_config.json" });
    const tts_json = try inference.models.qwen3_tts.testConfigJson(gpa, "1b7", 16, "custom_voice");
    defer gpa.free(tts_json);
    var diag: config.Diagnostic = .{};
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = config_path, .data = tts_json });
    try std.testing.expectEqual(catalog.ModelKind.speech, try speechCheckpoint(gpa, io, weights, &diag));
    // A greedy generation config is one `speak` runs, so it registers too.
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = generation_path, .data = "{\"do_sample\":false,\"max_new_tokens\":256}" });
    try std.testing.expectEqual(catalog.ModelKind.speech, try speechCheckpoint(gpa, io, weights, &diag));
    // A Base and a VoiceDesign register as speech; another type does not.
    const base_json = try inference.models.qwen3_tts.testConfigJson(gpa, "1b7", 16, "base");
    defer gpa.free(base_json);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = config_path, .data = base_json });
    try std.testing.expectEqual(catalog.ModelKind.speech, try speechCheckpoint(gpa, io, weights, &diag));
    const design_json = try inference.models.qwen3_tts.testConfigJson(gpa, "1b7", 16, "voice_design");
    defer gpa.free(design_json);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = config_path, .data = design_json });
    try std.testing.expectEqual(catalog.ModelKind.speech, try speechCheckpoint(gpa, io, weights, &diag));
    const other_json = try inference.models.qwen3_tts.testConfigJson(gpa, "1b7", 16, "voice_clone");
    defer gpa.free(other_json);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = config_path, .data = other_json });
    try std.testing.expectError(error.NotRunnable, speechCheckpoint(gpa, io, weights, &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message(), "UnsupportedModelType") != null);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = config_path, .data = "{\"model_type\":\"bert\"}" });
    try std.testing.expectError(error.NotRunnable, speechCheckpoint(gpa, io, weights, &diag));

    const jobs = [_]Job{
        .{ .name = "model.safetensors", .role = .main, .sha256 = null, .size = 1 },
        .{ .name = "config.json", .role = .support, .sha256 = null, .size = 1 },
        .{ .name = "speech_tokenizer/config.json", .role = .support, .sha256 = null, .size = 1 },
    };
    try std.testing.expect(besideWeights("model.safetensors", &jobs, "config.json"));
    try std.testing.expect(!besideWeights("model.safetensors", &jobs, "rl_agent_config.json"));
    try std.testing.expect(besideWeights("speech_tokenizer/model.safetensors", &jobs, "config.json"));
}

test "a verified file whose encoding this build does not store still gets its role from the request" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;
    var inventory = try inference.models.qwen35.inventoryDocument(gpa);
    defer inventory.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", arena);
    const path = try std.fs.path.join(arena, &.{ root, "ternary.gguf" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = try serializeDirectory(arena, inventory, .{ .encoding = .{ .tensor = "output.weight", .id = 34 } }) });
    try std.testing.expectEqual(@as(?Role, null), try roleFromHeader(gpa, io, path));
    try std.testing.expectEqual(Role.main, try resolveRole(try roleFromHeader(gpa, io, path), null));
    try std.testing.expectEqual(Role.mmproj, try resolveRole(try roleFromHeader(gpa, io, path), .mmproj));
}

test "inspect judges the pinned directory supported, runnable, or not, from the head of the file" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const entry = catalog.find("qwen3.8-27b").?;
    // The real inventory with the pinned digest: supported, assembled from
    // several windows, nothing read past the directory.
    const supported = try inspectFixture(arena, gpa, .{}, entry.main().sha256, 256 * 1024);
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
    const stored = try inspectFixture(arena, gpa, .{ .encoding = .{ .tensor = "token_embd.weight", .id = 2 } }, entry.main().sha256, 1 << 20);
    try std.testing.expectEqual(.not_runnable, stored.verdict.status);
    try std.testing.expectEqualStrings("token_embd.weight", stored.verdict.tensor.?);
    try std.testing.expectEqualStrings("Q4_0 (id 2)", stored.verdict.encoding.?);
    try std.testing.expect(std.mem.indexOf(u8, stored.verdict.reason, "outside the qwen35 adapter's executable set") != null);
    try std.testing.expect(stored.directory != null);
    // An encoding nuclis does not store: rejected by the parser, named anyway.
    const unknown = try inspectFixture(arena, gpa, .{ .encoding = .{ .tensor = "blk.0.ffn_up.weight", .id = 24 } }, entry.main().sha256, 1 << 20);
    try std.testing.expectEqual(.not_runnable, unknown.verdict.status);
    try std.testing.expectEqualStrings("blk.0.ffn_up.weight", unknown.verdict.tensor.?);
    try std.testing.expectEqualStrings("id 24", unknown.verdict.encoding.?);
    try std.testing.expect(std.mem.indexOf(u8, unknown.verdict.reason, "does not store") != null);
    try std.testing.expect(unknown.directory == null);
    // An architecture without an adapter.
    const foreign = try inspectFixture(arena, gpa, .{ .architecture = "gemma3" }, entry.main().sha256, 1 << 20);
    try std.testing.expectEqual(.not_runnable, foreign.verdict.status);
    try std.testing.expectEqualStrings("no adapter for architecture \"gemma3\"", foreign.verdict.reason);
    try std.testing.expect(foreign.verdict.tensor == null);
    // An architecture with an adapter that rejects this directory (the
    // Gemma adapter over the Qwen tensor list).
    const rejected = try inspectFixture(arena, gpa, .{ .architecture = "gemma4" }, entry.main().sha256, 1 << 20);
    try std.testing.expectEqual(.not_runnable, rejected.verdict.status);
    try std.testing.expect(std.mem.startsWith(u8, rejected.verdict.reason, "the gemma4 adapter rejects the file: "));
    // Both renderings carry the verdict.
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try renderInspect(&out.writer, supported, false, .none);
    try std.testing.expect(std.mem.startsWith(u8, out.written(), "unsloth/Qwen3.8-27B-GGUF @ main: commit 4ca720788d1e01f1bff70c033e0d0028fd02e502\nQwen3.8-27B-UD-Q4_K_M.gguf: "));
    try std.testing.expect(std.mem.indexOf(u8, out.written(), " bytes, sha256 322e194f") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "\nArchitecture: \"qwen35\"\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "\nVerdict: supported: index entry qwen3.8-27b") != null);
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
    var with: std.EnumSet(Role) = .empty;
    with.insert(.mmproj);
    const pull_options = try fromRegistry(.{ .repo = "gemma", .with = with, .force = true }, &gemma, &diag);
    try std.testing.expectEqualStrings("gemma", pull_options.name.?);
    try std.testing.expectEqualStrings("unsloth/gemma-4-12b-it-GGUF", pull_options.repo);
    try std.testing.expectEqualStrings("g.gguf", pull_options.file.?);
    try std.testing.expectEqualStrings("fc034cfff751157913579611efad8462ac1be606", pull_options.revision.?);
    try std.testing.expectEqualStrings("mmproj-F16.gguf", pull_options.mmproj.?);
    try std.testing.expect(pull_options.draft == null and pull_options.force and !pull_options.all and pull_options.with.count() == 0);
    const all = try fromRegistry(.{ .repo = "gemma", .all = true }, &gemma, &diag);
    try std.testing.expect(all.mmproj != null and all.draft == null and all.revision != null);
    const plain = try fromRegistry(.{ .repo = "gemma" }, &.{ .repo = "a/b", .file = "f.gguf" }, &diag);
    try std.testing.expect(plain.mmproj == null and plain.revision == null);
    with = .empty;
    with.insert(.draft);
    try std.testing.expectError(error.NoSuchCompanion, fromRegistry(.{ .repo = "gemma", .with = with }, &gemma, &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message(), "no draft companion") != null);
    // A draft variant is the index's, never a registry entry's.
    try std.testing.expectError(error.ConflictingOptions, fromRegistry(.{ .repo = "gemma", .with = with, .draft_variant = "dflash2" }, &.{ .repo = "a/b", .file = "f.gguf", .draft = "d.gguf" }, &diag));
    try std.testing.expectError(error.NotPullable, fromRegistry(.{ .repo = "local" }, &.{ .path = "/scratch/x.gguf" }, &diag));
    try std.testing.expectError(error.ConflictingOptions, fromRegistry(.{ .repo = "gemma", .file = "other.gguf" }, &gemma, &diag));
    try std.testing.expect(isRegistryName("gemma") and !isRegistryName("qwen3.8-27b") and !isRegistryName("embeddinggemma-2") and !isRegistryName("a/b") and !isRegistryName(""));
}

/// A file and, unless `role` is null, its sidecar, for the `rm` tests.
fn plant(arena: Allocator, io: std.Io, dir: std.Io.Dir, sub_path: []const u8, repo: []const u8, role: ?Role) !void {
    if (std.fs.path.dirname(sub_path)) |parent| try dir.createDirPath(io, parent);
    try dir.writeFile(io, .{ .sub_path = sub_path, .data = "0123456789" });
    const r = role orelse return;
    try writeSidecar(arena, io, dir, try sidecarPath(arena, sub_path), .{ .repo = repo, .file = sub_path, .revision = "0123456789abcdef0123456789abcdef01234567", .sha256 = "84d89877f0d4041efb6bf91a16f0248f2fd573e6af05c19f96bedb9f882f7882", .size = 10, .role = r, .downloaded_at = "2026-10-28T00:00:00Z", .nuclis_version = version });
}

test "rm removes an entry's files, sidecars, and support files, then the directories it emptied" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", alloc);
    defer alloc.free(root);
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diag: config.Diagnostic = .{};
    const laya = catalog.findKind("laya-multilingual", .decision).?;
    const base = "models/convaiinnovations/laya/";
    try plant(arena, io, tmp.dir, base ++ "multilingual/model.safetensors", laya.repo, .main);
    for (laya.main().support) |s| try plant(arena, io, tmp.dir, try std.mem.concat(arena, u8, &.{ "models/", laya.repo, "/", s }), laya.repo, .support);
    // A support file the catalogue does not name, found by its sidecar.
    try plant(arena, io, tmp.dir, base ++ "multilingual/eval/results.json", laya.repo, .support);
    // The English checkpoint's support file, its weights absent: a partly
    // present entry of its own, which the first removal leaves alone.
    try plant(arena, io, tmp.dir, base ++ "rl_agent_config.json", laya.repo, .support);
    try std.testing.expectError(error.NoSuchCompanion, remove(arena, io, root, .{}, "laya-multilingual", .{ .with = .initOne(.draft) }, &diag));

    const first = try remove(arena, io, root, .{}, "laya-multilingual", .{}, &diag);
    try std.testing.expectEqual(2 * (1 + laya.main().support.len + 1), first.removed.len);
    try std.testing.expectEqual(@as(usize, 0), first.kept.len);
    var freed: u64 = 0;
    for (first.removed) |r| freed += r.bytes;
    try std.testing.expectEqual(freed, first.freed_bytes);
    try std.testing.expectError(error.FileNotFound, tmp.dir.access(io, base ++ "multilingual", .{}));
    try tmp.dir.access(io, base ++ "rl_agent_config.json.nuclis.json", .{});

    const second = try remove(arena, io, root, .{}, "laya", .{}, &diag);
    try std.testing.expectEqual(@as(usize, 2), second.removed.len);
    try std.testing.expectError(error.FileNotFound, tmp.dir.access(io, "models/convaiinnovations", .{}));
    try tmp.dir.access(io, "models", .{});
    // Nothing left: an error naming the entry.
    try std.testing.expectError(error.NotPulled, remove(arena, io, root, .{}, "laya", .{}, &diag));
    try std.testing.expect(std.mem.startsWith(u8, diag.message(), "laya has nothing pulled"));
    try std.testing.expectError(error.NotPulled, remove(arena, io, root, .{}, "qwen3.8-27b", .{}, &diag));
    try std.testing.expectError(error.UnknownModel, remove(arena, io, root, .{}, "nothing-by-this-name", .{}, &diag));
}

test "rm keeps a file another entry uses, takes companions only when asked, and is not blocked by its registry twin" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", alloc);
    defer alloc.free(root);
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diag: config.Diagnostic = .{};
    const qwen = catalog.find("qwen3.8-27b").?;
    const main = try std.mem.concat(arena, u8, &.{ "models/", qwen.repo, "/", qwen.main().path });
    const mmproj = try std.mem.concat(arena, u8, &.{ "models/", qwen.repo, "/", qwen.companion(.mmproj).?.path });
    const mtp = try std.mem.concat(arena, u8, &.{ "models/", qwen.repo, "/", qwen.draft(null).?.path });
    try plant(arena, io, tmp.dir, main, qwen.repo, .main);
    try plant(arena, io, tmp.dir, mmproj, qwen.repo, .mmproj);
    try plant(arena, io, tmp.dir, mtp, qwen.repo, .draft);
    const twin: config.NamedModel = .{ .name = "qwen3.8-27b", .entry = config.registryEntry(qwen) };

    // Another name for the same file keeps it; the twin does not.
    const shared = [_]config.NamedModel{ twin, .{ .name = "big", .entry = .{ .repo = qwen.repo, .file = qwen.main().path } } };
    const blocked = try remove(arena, io, root, .{ .entries = &shared }, "qwen3.8-27b", .{}, &diag);
    try std.testing.expectEqual(@as(usize, 0), blocked.removed.len);
    try std.testing.expectEqual(@as(usize, 1), blocked.kept.len);
    try std.testing.expectEqualStrings("big", blocked.kept[0].used_by);
    try tmp.dir.access(io, main, .{});

    // The default leaves the companions; --with takes the one named.
    const registry: config.Models = .{ .entries = &.{twin} };
    const bare = try remove(arena, io, root, registry, "qwen3.8-27b", .{}, &diag);
    try std.testing.expectEqual(@as(usize, 2), bare.removed.len);
    try std.testing.expectError(error.FileNotFound, tmp.dir.access(io, main, .{}));
    try tmp.dir.access(io, mmproj, .{});
    try std.testing.expectError(error.NotPulled, remove(arena, io, root, registry, "qwen3.8-27b", .{}, &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message(), "--all") != null);
    const projector = try remove(arena, io, root, registry, "qwen3.8-27b", .{ .with = .initOne(.mmproj) }, &diag);
    try std.testing.expectEqual(@as(usize, 2), projector.removed.len);
    try tmp.dir.access(io, mtp, .{});
    _ = try remove(arena, io, root, registry, "qwen3.8-27b", .{ .all = true }, &diag);
    try std.testing.expectError(error.FileNotFound, tmp.dir.access(io, "models/unsloth", .{}));
    try tmp.dir.access(io, "models", .{});
}

test "rm by registry name: its file and named companions, never outside the models directory" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", alloc);
    defer alloc.free(root);
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diag: config.Diagnostic = .{};
    try plant(arena, io, tmp.dir, "models/acme/thing/x.gguf", "acme/thing", null);
    try plant(arena, io, tmp.dir, "models/acme/thing/mmproj.gguf", "acme/thing", .mmproj);
    try plant(arena, io, tmp.dir, "models/acme/thing/README.md", "acme/thing", null);
    const entries = [_]config.NamedModel{
        .{ .name = "local", .entry = .{ .path = "acme/thing/x.gguf", .mmproj = "mmproj.gguf" } },
        .{ .name = "away", .entry = .{ .path = "/elsewhere/x.gguf" } },
    };
    const registry: config.Models = .{ .entries = &entries };
    try std.testing.expectError(error.NoSuchCompanion, remove(arena, io, root, registry, "local", .{ .with = .initOne(.draft) }, &diag));
    const report = try remove(arena, io, root, registry, "local", .{ .with = .initOne(.mmproj) }, &diag);
    try std.testing.expectEqual(@as(usize, 3), report.removed.len);
    try std.testing.expectEqual(10 + 10 + report.removed[2].bytes, report.freed_bytes);
    // A file the removal did not name keeps the directory.
    try tmp.dir.access(io, "models/acme/thing/README.md", .{});
    try std.testing.expectError(error.OutsideModelsDirectory, remove(arena, io, root, registry, "away", .{}, &diag));
    // Both forms come from the same report.
    var text: std.Io.Writer.Allocating = .init(alloc);
    defer text.deinit();
    try report.render(&text.writer, false, .none);
    try std.testing.expect(std.mem.indexOf(u8, text.written(), "removed acme/thing/x.gguf") != null);
    var json: std.Io.Writer.Allocating = .init(alloc);
    defer json.deinit();
    try report.render(&json.writer, true, .none);
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, json.written(), .{});
    defer parsed.deinit();
    try std.testing.expectEqual(@as(i64, @intCast(report.freed_bytes)), parsed.value.object.get("freed_bytes").?.integer);
    try std.testing.expectEqual(@as(usize, 3), parsed.value.object.get("removed").?.array.items.len);
}

test "adopt writes a sidecar for a file placed by hand, once, and only under the models directory" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", alloc);
    defer alloc.free(root);
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diag: config.Diagnostic = .{};
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    try plant(arena, io, tmp.dir, "models/acme/encoder/model.safetensors", "acme/encoder", null);
    // A file the index does not pin keeps its local digest and no commit.
    try adopt(alloc, io, root, "acme/encoder/model.safetensors", false, true, &out.writer, .none, &diag);
    const report = try std.json.parseFromSlice(std.json.Value, alloc, out.written(), .{});
    defer report.deinit();
    try std.testing.expect(report.value.object.get("artifact").? == .null);
    try std.testing.expectEqualStrings("84d89877f0d4041efb6bf91a16f0248f2fd573e6af05c19f96bedb9f882f7882", report.value.object.get("sha256").?.string);
    const written = (try readSidecar(arena, io, tmp.dir, "models/acme/encoder/model.safetensors.nuclis.json")).?;
    try std.testing.expect(written.revision == null and written.downloaded_at == null and written.adopted_at != null);
    try std.testing.expectEqualStrings("acme/encoder", written.repo);
    try std.testing.expectEqualStrings("model.safetensors", written.file);
    try std.testing.expectEqual(Role.main, written.role);
    // The listing no longer counts it as unverified.
    try std.testing.expectEqual(@as(usize, 0), (try list(arena, io, root, .{}, null)).without_sidecar.len);
    // Once only, unless forced.
    try std.testing.expectError(error.AlreadyAdopted, adopt(alloc, io, root, "acme/encoder/model.safetensors", false, true, &out.writer, .none, &diag));
    try adopt(alloc, io, root, "acme/encoder/model.safetensors", true, true, &out.writer, .none, &diag);
    // Nothing outside the models directory or its layout, nothing that is not a model file.
    try std.testing.expectError(error.OutsideModelsDirectory, adopt(alloc, io, root, "../nuclis.json", false, true, &out.writer, .none, &diag));
    try plant(arena, io, tmp.dir, "models/loose.gguf", "", null);
    try std.testing.expectError(error.OutsideModelsDirectory, adopt(alloc, io, root, "loose.gguf", false, true, &out.writer, .none, &diag));
    try plant(arena, io, tmp.dir, "models/acme/encoder/notes.txt", "acme/encoder", null);
    try std.testing.expectError(error.NotAModelFile, adopt(alloc, io, root, "acme/encoder/notes.txt", false, true, &out.writer, .none, &diag));
    // A GGUF the index does not pin must parse before anything is written.
    try plant(arena, io, tmp.dir, "models/acme/encoder/broken.gguf", "acme/encoder", null);
    try std.testing.expectError(error.NotAModelFile, adopt(alloc, io, root, "acme/encoder/broken.gguf", false, true, &out.writer, .none, &diag));
    try std.testing.expectError(error.FileNotFound, tmp.dir.access(io, "models/acme/encoder/broken.gguf.nuclis.json", .{}));
    try std.testing.expectError(error.FileNotFound, adopt(alloc, io, root, "acme/encoder/absent.gguf", false, true, &out.writer, .none, &diag));
}

test "a pull takes each role's default draft, a named variant, or every one, grouped by origin" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diag: config.Diagnostic = .{};
    const qwen = catalog.find("qwen3.8-27b").?;
    const Paths = struct {
        fn of(a: Allocator, files: []const *const catalog.File) ![]const u8 {
            var out: std.ArrayList(u8) = .empty;
            for (files) |f| {
                if (out.items.len > 0) try out.append(a, ' ');
                try out.appendSlice(a, if (f.variant) |v| v else @tagName(f.role));
            }
            return out.items;
        }
    };
    try std.testing.expectEqualStrings("main", try Paths.of(arena, try selectFiles(arena, qwen, .{}, &diag)));
    try std.testing.expectEqualStrings("main mtp", try Paths.of(arena, try selectFiles(arena, qwen, .{ .with = .initOne(.draft) }, &diag)));
    try std.testing.expectEqualStrings("main dflash2", try Paths.of(arena, try selectFiles(arena, qwen, .{ .with = .initOne(.draft), .draft_variant = "dflash2" }, &diag)));
    try std.testing.expectEqualStrings("main mmproj mtp", try Paths.of(arena, try selectFiles(arena, qwen, .{ .all = true }, &diag)));
    const every = try selectFiles(arena, qwen, .{ .all = true, .variants = true }, &diag);
    try std.testing.expectEqualStrings("main mmproj mtp dflash2", try Paths.of(arena, every));
    try std.testing.expectError(error.NoSuchCompanion, selectFiles(arena, qwen, .{ .with = .initOne(.draft), .draft_variant = "eagle" }, &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message(), "it has: mtp, dflash2") != null);
    try std.testing.expectError(error.NoSuchCompanion, selectFiles(arena, catalog.find("laya").?, .{ .with = .initOne(.draft) }, &diag));
    // The drafter from z-lab's repository is its own plan, after the model's own.
    const groups = try groupByOrigin(arena, qwen, every);
    try std.testing.expectEqual(@as(usize, 2), groups.len);
    try std.testing.expectEqualStrings(qwen.repo, groups[0].repo);
    try std.testing.expectEqual(@as(usize, 3), groups[0].files.len);
    try std.testing.expectEqualStrings("z-lab/Qwen3.8-27B-DFlash2-GGUF", groups[1].repo);
    try std.testing.expectEqualStrings("2d9571f8ce46e151f61c6499c99dee6079e1d610", groups[1].revision);
    // It lands beside the model it drafts for.
    const local = try catalog.localPath(arena, "/m", qwen, groups[1].files[0]);
    try std.testing.expectEqualStrings("/m/unsloth/Qwen3.8-27B-GGUF/MTP/Qwen3.8-27B-DFlash2-Q8_0.gguf", local);
    try std.testing.expectEqualStrings("Qwen3.8-27B-DFlash2-Q8_0.gguf", catalog.Artifact.hubName(groups[1].files[0]));
}
