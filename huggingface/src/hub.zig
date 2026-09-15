//! Hub identity and selection are separate from transfer. Every catalog pins a
//! branch/tag to a commit before a byte of model data is downloaded.
const std = @import("std");
const http = @import("http.zig");
pub const Allocator = std.mem.Allocator;

pub const Request = struct {
    repo_id: []const u8,
    filename: ?[]const u8 = null,
    revision: []const u8 = "main",
    /// Absolute or relative caller-supplied directory; no implicit '~' expansion.
    local_dir: ?[]const u8 = null,
};

pub const File = struct {
    name: []const u8,
    size: u64,
    sha256: [32]u8,
};

pub const Catalog = struct {
    arena: std.heap.ArenaAllocator,
    repo_id: []const u8,
    revision: [40]u8,
    files: []const File,
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
    if (req.filename) |f| if (!validPath(f) or !std.mem.endsWith(u8, f, ".gguf")) return error.InvalidFilename;
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
    for (w.siblings) |f| {
        if (!std.mem.endsWith(u8, f.rfilename, ".gguf")) continue;
        if (!validPath(f.rfilename)) return error.InvalidMetadata;
        const lfs = f.lfs orelse return error.MissingChecksum;
        if (lfs.sha256.len != 64 or !isHex(lfs.sha256)) return error.InvalidMetadata;
        if (f.size) |n| if (n != lfs.size) return error.InvalidMetadata;
        if (lfs.size < 4 or lfs.size > 512 * 1024 * 1024 * 1024) return error.InvalidMetadata;
        for (files.items) |old| if (std.mem.eql(u8, old.name, f.rfilename)) return error.InvalidMetadata;
        var sha: [32]u8 = undefined;
        _ = try std.fmt.hexToBytes(&sha, lfs.sha256);
        try files.append(a, .{ .name = f.rfilename, .size = lfs.size, .sha256 = sha });
    }
    return .{ .arena = arena, .repo_id = try a.dupe(u8, repo), .revision = w.sha[0..40].*, .files = files.items };
}

pub fn isHex(s: []const u8) bool {
    for (s) |b| if (!std.ascii.isHex(b)) return false;
    return true;
}

const Shard = struct { prefix: []const u8, index: u32, total: u32 };
fn shard(name: []const u8) ?Shard {
    // GGUF standard split filename: name-00001-of-00002.gguf.
    if (name.len < 20) return null;
    const suffix = name[name.len - 20 ..];
    if (suffix[0] != '-' or !std.mem.eql(u8, suffix[6..10], "-of-") or !std.mem.eql(u8, suffix[15..], ".gguf")) return null;
    const index = std.fmt.parseInt(u32, suffix[1..6], 10) catch return null;
    const total = std.fmt.parseInt(u32, suffix[10..15], 10) catch return null;
    return .{ .prefix = name[0 .. name.len - 20], .index = index, .total = total };
}

/// null means the caller must choose a quantization. A split-file selection
/// expands to its complete ordered shard set; missing or duplicate shards fail.
pub fn select(a: Allocator, files: []const File, filename: ?[]const u8) !?[]const File {
    if (files.len == 0) return error.NoGgufFiles;
    var chosen: ?File = null;
    if (filename) |name| {
        for (files) |f| if (std.mem.eql(u8, name, f.name)) {
            chosen = f;
            break;
        };
        if (chosen == null) return error.FileNotFound;
    } else {
        chosen = files[0];
        const first = shard(files[0].name);
        if (files.len > 1) {
            const s = first orelse return null;
            for (files[1..]) |f| {
                const other = shard(f.name) orelse return null;
                if (!std.mem.eql(u8, s.prefix, other.prefix) or s.total != other.total) return null;
            }
        }
    }
    const f = chosen.?;
    if (shard(f.name)) |s| {
        if (s.total == 0 or s.total > 10_000 or s.index == 0 or s.index > s.total) return error.InvalidShardSet;
        const result = try a.alloc(File, s.total);
        errdefer a.free(result);
        for (1..@as(usize, s.total) + 1) |index| {
            var found: ?File = null;
            for (files) |part| if (shard(part.name)) |p| {
                if (std.mem.eql(u8, p.prefix, s.prefix) and p.total == s.total and p.index == index) {
                    if (found != null) return error.InvalidShardSet;
                    found = part;
                }
            };
            result[index - 1] = found orelse return error.InvalidShardSet;
        }
        return result;
    }
    return try a.dupe(File, &.{f});
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

test "exact selection, ambiguity, and split set" {
    const a = std.testing.allocator;
    const files = [_]File{
        .{ .name = "Q4.gguf", .size = 4, .sha256 = @splat(0) },
        .{ .name = "Q8.gguf", .size = 4, .sha256 = @splat(0) },
    };
    try std.testing.expect(try select(a, &files, null) == null);
    try std.testing.expectError(error.FileNotFound, select(a, &files, "4.gguf"));
    const picked = (try select(a, &files, "Q4.gguf")).?;
    defer a.free(picked);
    try std.testing.expectEqualStrings("Q4.gguf", picked[0].name);
    const parts = [_]File{
        .{ .name = "Q4-00002-of-00002.gguf", .size = 4, .sha256 = @splat(0) },
        .{ .name = "Q4-00001-of-00002.gguf", .size = 4, .sha256 = @splat(0) },
    };
    const selected = (try select(a, &parts, null)).?;
    defer a.free(selected);
    try std.testing.expectEqualStrings(parts[1].name, selected[0].name);
    try std.testing.expectError(error.InvalidShardSet, select(a, parts[0..1], parts[0].name));
}

const catalog_fixture =
    \\{"sha":"0123456789abcdef0123456789abcdef01234567","siblings":[{"rfilename":"Q4.gguf","size":8,"lfs":{"size":8,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}},{"rfilename":"README.md"}]}
;
fn catalogRoundTrip(a: Allocator) !void {
    var c = try parseCatalog(a, "a/b", catalog_fixture);
    defer c.deinit();
    try std.testing.expectEqual(@as(usize, 1), c.files.len);
    try std.testing.expectEqualStrings("Q4.gguf", c.files[0].name);
}
test "catalog owns all strings and cleans up allocation failures" {
    try catalogRoundTrip(std.testing.allocator);
    try std.testing.checkAllAllocationFailures(std.testing.allocator, catalogRoundTrip, .{});
    const bad = try std.mem.replaceOwned(u8, std.testing.allocator, catalog_fixture, "Q4.gguf", "../Q4.gguf");
    defer std.testing.allocator.free(bad);
    try std.testing.expectError(error.InvalidMetadata, parseCatalog(std.testing.allocator, "a/b", bad));
}
