//! Hub identity and selection are separate from transfer. Every catalog pins a
//! branch/tag to a commit before a byte of model data is downloaded.
//!
//! An artifact is a GGUF file, a safetensors file, or a standard
//! `-NNNNN-of-MMMMM` shard set of either. A safetensors artifact also takes
//! its support files (configuration, tokenizer, index) from its directory,
//! so the set is loadable on its own.
const std = @import("std");
const http = @import("http.zig");
pub const Allocator = std.mem.Allocator;

pub const Request = struct {
    repo_id: []const u8,
    filename: ?[]const u8 = null,
    revision: []const u8 = "main",
    /// Absolute or relative caller-supplied directory; no implicit '~' expansion.
    local_dir: ?[]const u8 = null,
    /// Download only `filename`, which may be any listed file (a support
    /// file included), without shard or support-file expansion.
    exact: bool = false,
};

pub const Format = enum { gguf, safetensors };

/// The weight container a filename names, by extension.
pub fn format(name: []const u8) ?Format {
    if (std.mem.endsWith(u8, name, ".gguf")) return .gguf;
    if (std.mem.endsWith(u8, name, ".safetensors")) return .safetensors;
    return null;
}

/// Extensions a safetensors artifact carries beside its weights. Code and
/// pickled weights (`.py`, `.bin`, `.pt`) are never selected.
const support_extensions = [_][]const u8{ ".json", ".txt", ".model", ".jinja" };

pub const File = struct {
    name: []const u8,
    size: u64,
    /// The content digest the Hub records for LFS/Xet files.
    sha256: ?[32]u8 = null,
    /// The git blob id (`sha1("blob <size>\0" ++ bytes)`) of a plain git
    /// file, which has no SHA-256 on the Hub. Exactly one of the two is set.
    git_oid: ?[20]u8 = null,
};

pub const Catalog = struct {
    arena: std.heap.ArenaAllocator,
    repo_id: []const u8,
    revision: [40]u8,
    /// Weight files (`.gguf`, `.safetensors`): the choices a request selects from.
    files: []const File,
    /// Files a safetensors artifact may carry (see `support_extensions`).
    support: []const File = &.{},
    pub fn deinit(c: *Catalog) void {
        c.arena.deinit();
        c.* = undefined;
    }
};

pub fn validPath(path: []const u8) bool {
    if (path.len == 0 or path.len > 4096 or std.mem.indexOfAny(u8, path, "\\\x00\r\n:") != null) return false;
    for (path) |b| if (b < 32 or b == 127) return false;
    var it = std.mem.splitScalar(u8, path, '/');
    while (it.next()) |part| if (part.len == 0 or std.mem.eql(u8, part, ".") or std.mem.eql(u8, part, "..")) return false;
    return true;
}

pub fn validate(req: Request) !void {
    if (!validPath(req.repo_id) or std.mem.count(u8, req.repo_id, "/") != 1) return error.InvalidRepository;
    for (req.repo_id) |b| if (!(std.ascii.isAlphanumeric(b) or std.mem.indexOfScalar(u8, "-._/", b) != null)) return error.InvalidRepository;
    if (!validPath(req.revision)) return error.InvalidRevision;
    if (req.filename) |f| if (!validPath(f) or (!req.exact and format(f) == null)) return error.InvalidFilename;
    if (req.exact and req.filename == null) return error.FilenameRequired;
    if (req.local_dir) |dir| if (dir.len == 0 or std.mem.indexOfScalar(u8, dir, 0) != null) return error.InvalidDirectory;
}

/// URL encoding treats a revision as one component, including branches with '/'.
/// File and repository slashes are retained only after traversal validation.
pub fn encode(a: Allocator, value: []const u8, keep_slashes: bool) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    const hex = "0123456789ABCDEF";
    for (value) |b| {
        if (std.ascii.isAlphanumeric(b) or std.mem.indexOfScalar(u8, "-._~", b) != null or (keep_slashes and b == '/')) {
            try out.append(a, b);
        } else try out.appendSlice(a, &.{ '%', hex[b >> 4], hex[b & 15] });
    }
    return out.toOwnedSlice(a);
}

pub fn directory(a: Allocator, explicit: ?[]const u8, nuclis_home: ?[]const u8, home: ?[]const u8) ![]const u8 {
    if (explicit) |p| {
        if (p.len == 0 or std.mem.indexOfScalar(u8, p, 0) != null) return error.InvalidDirectory;
        return a.dupe(u8, p);
    }
    if (nuclis_home) |p| {
        if (!std.fs.path.isAbsolute(p) or std.mem.indexOfScalar(u8, p, 0) != null) return error.InvalidNuclisHome;
        return std.fs.path.join(a, &.{ p, "models" });
    }
    const p = home orelse return error.MissingHome;
    if (!std.fs.path.isAbsolute(p) or std.mem.indexOfScalar(u8, p, 0) != null) return error.MissingHome;
    return std.fs.path.join(a, &.{ p, ".nuclis", "models" });
}

/// Hub replies to the model API and `resolve`: a 401 without a token means
/// the repository needs one (`TokenRequired`), with a token that the token was
/// refused (`TokenRejected`); a 403 means the token lacks access, typically a
/// gated repository whose terms were not accepted on the Hub (`AccessDenied`).
pub fn hubStatus(status: u16, token: ?[]const u8) !void {
    switch (status) {
        401 => return if (token == null) error.TokenRequired else error.TokenRejected,
        403 => return error.AccessDenied,
        else => return http.statusError(status),
    }
}

/// An `HF_TOKEN` value worth sending: null when unset or blank.
pub fn tokenFromEnvironment(value: ?[]const u8) ?[]const u8 {
    const v = std.mem.trim(u8, value orelse return null, " \t\r\n");
    return if (v.len == 0) null else v;
}

pub fn catalog(gpa: Allocator, io: std.Io, token: ?[]const u8, req: Request) !Catalog {
    try validate(req);
    var temp = std.heap.ArenaAllocator.init(gpa);
    defer temp.deinit();
    const a = temp.allocator();
    const url = try std.fmt.allocPrint(a, "https://huggingface.co/api/models/{s}/revision/{s}?blobs=true", .{
        req.repo_id, try encode(a, req.revision, false),
    });
    var client = http.client(gpa, io);
    defer client.deinit();
    var response = try http.get(&client, .{ .url = url, .token = token });
    defer response.deinit();
    try hubStatus(response.status, token);
    return parseCatalog(gpa, req.repo_id, response.body);
}

pub fn parseCatalog(gpa: Allocator, repo: []const u8, bytes: []const u8) !Catalog {
    const Wire = struct {
        sha: []const u8,
        siblings: []const struct {
            rfilename: []const u8,
            size: ?u64 = null,
            blobId: ?[]const u8 = null,
            lfs: ?struct { sha256: []const u8, size: u64 } = null,
        },
    };
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const parsed = try std.json.parseFromSlice(Wire, a, bytes, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
    const w = parsed.value;
    if (w.sha.len != 40 or !isHex(w.sha)) return error.InvalidMetadata;
    if (w.siblings.len > 10_000) return error.TooManyFiles;
    var files: std.ArrayList(File) = .empty;
    var support: std.ArrayList(File) = .empty;
    for (w.siblings) |f| {
        const kind = format(f.rfilename);
        const list = if (kind != null) &files else if (isSupport(f.rfilename)) &support else continue;
        if (!validPath(f.rfilename)) return error.InvalidMetadata;
        var file: File = .{ .name = f.rfilename, .size = undefined };
        if (f.lfs) |lfs| {
            if (lfs.sha256.len != 64 or !isHex(lfs.sha256)) return error.InvalidMetadata;
            if (f.size) |n| if (n != lfs.size) return error.InvalidMetadata;
            file.size = lfs.size;
            file.sha256 = undefined;
            _ = try std.fmt.hexToBytes(&file.sha256.?, lfs.sha256);
        } else {
            // A GGUF is always LFS on the Hub; a plain one is not a model.
            if (kind == .gguf) return error.MissingChecksum;
            const oid = f.blobId orelse return error.MissingChecksum;
            if (oid.len != 40 or !isHex(oid)) return error.InvalidMetadata;
            file.size = f.size orelse return error.InvalidMetadata;
            file.git_oid = undefined;
            _ = try std.fmt.hexToBytes(&file.git_oid.?, oid);
        }
        // The smallest valid containers: the GGUF magic, a safetensors
        // length prefix and `{}`.
        const minimum: u64 = if (kind) |k| switch (k) {
            .gguf => 4,
            .safetensors => 10,
        } else 0;
        if (file.size < minimum or file.size > 512 * 1024 * 1024 * 1024) return error.InvalidMetadata;
        for (list.items) |old| if (std.mem.eql(u8, old.name, f.rfilename)) return error.InvalidMetadata;
        try list.append(a, file);
    }
    return .{ .arena = arena, .repo_id = try a.dupe(u8, repo), .revision = w.sha[0..40].*, .files = files.items, .support = support.items };
}

fn isSupport(name: []const u8) bool {
    for (support_extensions) |ext| if (std.mem.endsWith(u8, name, ext)) return true;
    return false;
}

pub fn isHex(s: []const u8) bool {
    for (s) |b| if (!std.ascii.isHex(b)) return false;
    return true;
}

const Shard = struct { prefix: []const u8, index: u32, total: u32, format: Format };
/// The standard split filename, `name-00001-of-00002.gguf` (or `.safetensors`).
fn shard(name: []const u8) ?Shard {
    const kind = format(name) orelse return null;
    const extension = if (kind == .gguf) ".gguf" else ".safetensors";
    const stem = name[0 .. name.len - extension.len];
    if (stem.len < 15) return null;
    const suffix = stem[stem.len - 15 ..];
    if (suffix[0] != '-' or !std.mem.eql(u8, suffix[6..10], "-of-")) return null;
    const index = std.fmt.parseInt(u32, suffix[1..6], 10) catch return null;
    const total = std.fmt.parseInt(u32, suffix[10..15], 10) catch return null;
    return .{ .prefix = stem[0 .. stem.len - 15], .index = index, .total = total, .format = kind };
}

/// Whether two weight files belong to one artifact: the same file, or
/// shards of one set.
fn sameArtifact(a: []const u8, b: []const u8) bool {
    if (std.mem.eql(u8, a, b)) return true;
    const x = shard(a) orelse return false;
    const y = shard(b) orelse return false;
    return x.format == y.format and x.total == y.total and std.mem.eql(u8, x.prefix, y.prefix);
}

/// The weights and support files a request downloads, in download order:
/// a shard set in index order, then (for safetensors) the support files.
/// null means the repository holds several artifacts and the caller must
/// choose one; missing or duplicate shards fail.
pub fn select(a: Allocator, catalog_: *const Catalog, filename: ?[]const u8) !?[]const File {
    const files = catalog_.files;
    if (files.len == 0) return error.NoModelFiles;
    var chosen: ?File = null;
    if (filename) |name| {
        for (files) |f| if (std.mem.eql(u8, name, f.name)) {
            chosen = f;
            break;
        };
        if (chosen == null) return error.FileNotFound;
    } else {
        chosen = files[0];
        for (files[1..]) |f| if (!sameArtifact(files[0].name, f.name)) return null;
    }
    const f = chosen.?;
    var result: std.ArrayList(File) = .empty;
    errdefer result.deinit(a);
    if (shard(f.name)) |s| {
        if (s.total == 0 or s.total > 10_000 or s.index == 0 or s.index > s.total) return error.InvalidShardSet;
        for (1..@as(usize, s.total) + 1) |index| {
            var found: ?File = null;
            for (files) |part| if (shard(part.name)) |p| {
                if (p.format == s.format and std.mem.eql(u8, p.prefix, s.prefix) and p.total == s.total and p.index == index) {
                    if (found != null) return error.InvalidShardSet;
                    found = part;
                }
            };
            try result.append(a, found orelse return error.InvalidShardSet);
        }
    } else try result.append(a, f);
    if (format(f.name) == .safetensors) {
        const set_stem = if (shard(f.name)) |s| s.prefix else f.name[0 .. f.name.len - ".safetensors".len];
        for (catalog_.support) |file| if (supports(files, set_stem, file.name)) try result.append(a, file);
    }
    return try result.toOwnedSlice(a);
}

/// Whether a support file belongs to the safetensors set whose shards are
/// `<set_stem>...`: it lies in the set's directory or below, but not below
/// a subdirectory holding other safetensors weights (another artifact),
/// and an index file only goes with the set it indexes.
fn supports(files: []const File, set_stem: []const u8, name: []const u8) bool {
    const set_dir = std.fs.path.dirnamePosix(set_stem) orelse "";
    const below = if (set_dir.len == 0) name else blk: {
        if (!std.mem.startsWith(u8, name, set_dir) or name.len <= set_dir.len or name[set_dir.len] != '/') return false;
        break :blk name[set_dir.len + 1 ..];
    };
    if (std.mem.endsWith(u8, name, ".safetensors.index.json"))
        return std.mem.eql(u8, name[0 .. name.len - ".safetensors.index.json".len], set_stem);
    // Each subdirectory between the set's directory and the file.
    var end = std.mem.indexOfScalar(u8, below, '/');
    while (end) |e| : (end = if (std.mem.indexOfScalarPos(u8, below, e + 1, '/')) |n| n else null) {
        const sub = name[0 .. name.len - below.len + e];
        for (files) |w| if (format(w.name) == .safetensors) {
            if (std.mem.eql(u8, std.fs.path.dirnamePosix(w.name) orelse "", sub)) return false;
        };
    }
    return true;
}

test "hub status maps authorization failures by whether a token was sent" {
    try std.testing.expectError(error.TokenRequired, hubStatus(401, null));
    try std.testing.expectError(error.TokenRejected, hubStatus(401, "hf_x"));
    try std.testing.expectError(error.AccessDenied, hubStatus(403, "hf_x"));
    try std.testing.expectError(error.NotFound, hubStatus(404, null));
    try std.testing.expect(tokenFromEnvironment(null) == null);
    try std.testing.expect(tokenFromEnvironment("  ") == null);
    try std.testing.expectEqualStrings("hf_x", tokenFromEnvironment(" hf_x\n").?);
}

test "request and directory validation" {
    const a = std.testing.allocator;
    try validate(.{ .repo_id = "unsloth/Qwen3.8-27B-GGUF", .revision = "refs/pr/1" });
    try std.testing.expectError(error.InvalidRepository, validate(.{ .repo_id = "../escape" }));
    try std.testing.expectError(error.InvalidFilename, validate(.{ .repo_id = "a/b", .filename = "../bad.gguf" }));
    try std.testing.expectError(error.InvalidNuclisHome, directory(a, null, "relative", "/home/me"));
    const p = try directory(a, null, "/data/nuclis", null);
    defer a.free(p);
    try std.testing.expectEqualStrings("/data/nuclis/models", p);
    const q = try directory(a, "local", "invalid", null);
    defer a.free(q);
    try std.testing.expectEqualStrings("local", q);
    const e = try encode(a, "refs/pr/1", false);
    defer a.free(e);
    try std.testing.expectEqualStrings("refs%2Fpr%2F1", e);
}

/// A catalog over borrowed test slices; nothing to free.
fn fixtureCatalog(files: []const File, support: []const File) Catalog {
    return .{ .arena = undefined, .repo_id = "a/b", .revision = @splat('0'), .files = files, .support = support };
}

fn expectNames(expected: []const []const u8, actual: []const File) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, f| try std.testing.expectEqualStrings(e, f.name);
}

test "exact selection, ambiguity, and split set" {
    const a = std.testing.allocator;
    const files = [_]File{
        .{ .name = "Q4.gguf", .size = 4, .sha256 = @splat(0) },
        .{ .name = "Q8.gguf", .size = 4, .sha256 = @splat(0) },
    };
    try std.testing.expect(try select(a, &fixtureCatalog(&files, &.{}), null) == null);
    try std.testing.expectError(error.FileNotFound, select(a, &fixtureCatalog(&files, &.{}), "4.gguf"));
    const picked = (try select(a, &fixtureCatalog(&files, &.{}), "Q4.gguf")).?;
    defer a.free(picked);
    try std.testing.expectEqualStrings("Q4.gguf", picked[0].name);
    const parts = [_]File{
        .{ .name = "Q4-00002-of-00002.gguf", .size = 4, .sha256 = @splat(0) },
        .{ .name = "Q4-00001-of-00002.gguf", .size = 4, .sha256 = @splat(0) },
    };
    const selected = (try select(a, &fixtureCatalog(&parts, &.{}), null)).?;
    defer a.free(selected);
    try std.testing.expectEqualStrings(parts[1].name, selected[0].name);
    try std.testing.expectError(error.InvalidShardSet, select(a, &fixtureCatalog(parts[0..1], &.{}), parts[0].name));
    try std.testing.expectError(error.NoModelFiles, select(a, &fixtureCatalog(&.{}, &.{}), null));
}

test "a safetensors artifact takes its shards and its own support files" {
    const a = std.testing.allocator;
    // Laya's layout: three sets, one at the root and two in subdirectories.
    const laya = [_]File{
        .{ .name = "model.safetensors", .size = 10 },
        .{ .name = "multilingual/model.safetensors", .size = 10 },
        .{ .name = "typed-decisions/model.safetensors", .size = 10 },
    };
    const laya_support = [_]File{
        .{ .name = "encoder/config.json", .size = 1 },
        .{ .name = "eval/results.json", .size = 1 },
        .{ .name = "multilingual/encoder/config.json", .size = 1 },
        .{ .name = "multilingual/tokenizer/tokenizer.json", .size = 1 },
        .{ .name = "rl_agent_config.json", .size = 1 },
        .{ .name = "tokenizer/tokenizer.json", .size = 1 },
        .{ .name = "typed-decisions/rl_agent_config.json", .size = 1 },
    };
    const catalog_ = fixtureCatalog(&laya, &laya_support);
    try std.testing.expect(try select(a, &catalog_, null) == null);
    const root = (try select(a, &catalog_, "model.safetensors")).?;
    defer a.free(root);
    try expectNames(&.{ "model.safetensors", "encoder/config.json", "eval/results.json", "rl_agent_config.json", "tokenizer/tokenizer.json" }, root);
    const multilingual = (try select(a, &catalog_, "multilingual/model.safetensors")).?;
    defer a.free(multilingual);
    try expectNames(&.{ "multilingual/model.safetensors", "multilingual/encoder/config.json", "multilingual/tokenizer/tokenizer.json" }, multilingual);

    // A sharded set beside a consolidated file (Mistral's layout): two
    // artifacts, and each index file goes with the set it indexes.
    const mixed = [_]File{
        .{ .name = "consolidated.safetensors", .size = 10 },
        .{ .name = "model-00002-of-00002.safetensors", .size = 10 },
        .{ .name = "model-00001-of-00002.safetensors", .size = 10 },
    };
    const mixed_support = [_]File{
        .{ .name = "config.json", .size = 1 },
        .{ .name = "model.safetensors.index.json", .size = 1 },
    };
    const mixed_catalog = fixtureCatalog(&mixed, &mixed_support);
    try std.testing.expect(try select(a, &mixed_catalog, null) == null);
    const sharded = (try select(a, &mixed_catalog, "model-00002-of-00002.safetensors")).?;
    defer a.free(sharded);
    try expectNames(&.{ "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "config.json", "model.safetensors.index.json" }, sharded);
    const consolidated = (try select(a, &mixed_catalog, "consolidated.safetensors")).?;
    defer a.free(consolidated);
    try expectNames(&.{ "consolidated.safetensors", "config.json" }, consolidated);
    // The sole artifact is selected without a name; a GGUF takes no support files.
    const sole = (try select(a, &fixtureCatalog(mixed[1..], &mixed_support), null)).?;
    defer a.free(sole);
    try std.testing.expectEqual(@as(usize, 4), sole.len);
    const gguf = (try select(a, &fixtureCatalog(&.{.{ .name = "m.gguf", .size = 4, .sha256 = @splat(0) }}, &mixed_support), null)).?;
    defer a.free(gguf);
    try expectNames(&.{"m.gguf"}, gguf);
}

const catalog_fixture =
    \\{"sha":"0123456789abcdef0123456789abcdef01234567","siblings":[{"rfilename":"Q4.gguf","size":8,"lfs":{"size":8,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}},{"rfilename":"README.md"},
    \\{"rfilename":"config.json","size":704,"blobId":"5881ba831f2db5ce0f606bbaa1f2668e1e6cb706"},{"rfilename":"model.py","size":10,"blobId":"5881ba831f2db5ce0f606bbaa1f2668e1e6cb706"},
    \\{"rfilename":"model.safetensors","size":16,"blobId":"5881ba831f2db5ce0f606bbaa1f2668e1e6cb706","lfs":{"size":16,"sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}},{"rfilename":"logo.png","size":10,"lfs":{"size":10,"sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}}]}
;
fn catalogRoundTrip(a: Allocator) !void {
    var c = try parseCatalog(a, "a/b", catalog_fixture);
    defer c.deinit();
    try expectNames(&.{ "Q4.gguf", "model.safetensors" }, c.files);
    try expectNames(&.{"config.json"}, c.support);
    // An LFS file is verified by its SHA-256, a plain one by its git blob id.
    try std.testing.expect(c.files[1].sha256 != null and c.files[1].git_oid == null);
    try std.testing.expect(c.support[0].sha256 == null and c.support[0].git_oid.?[0] == 0x58);
}
test "catalog owns all strings and cleans up allocation failures" {
    try catalogRoundTrip(std.testing.allocator);
    try std.testing.checkAllAllocationFailures(std.testing.allocator, catalogRoundTrip, .{});
    const bad = try std.mem.replaceOwned(u8, std.testing.allocator, catalog_fixture, "Q4.gguf", "../Q4.gguf");
    defer std.testing.allocator.free(bad);
    try std.testing.expectError(error.InvalidMetadata, parseCatalog(std.testing.allocator, "a/b", bad));
}
