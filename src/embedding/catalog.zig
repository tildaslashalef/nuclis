//! Embedding models by name: resolve a name to its GGUF file and projector
//! (the registry first, then the embedding catalogue, then a path), and the
//! identity a vector's space is named from. Shared by `nuclis embed` and the
//! API; the names of language and decision models are refused.
const std = @import("std");
const config = @import("../config.zig");
const catalog = @import("../catalog.zig");
const model = @import("../model.zig");
const response = @import("response.zig");

/// Where a name's files are, before anything is checked on disk.
pub const Resolved = struct {
    path: []const u8,
    /// The projector the entry names, pulled or not.
    mmproj: ?[]const u8 = null,
    entry: ?*const catalog.EmbeddingEntry = null,
};

/// The file `name` names, in order: a registry entry (which must be of kind
/// `embedding`), an embedding catalogue name, else a path (as given when
/// absolute or `./`, else under `<root>/models`). Another kind's name is
/// `NotAnEmbeddingModel`.
pub fn resolve(arena: std.mem.Allocator, root: ?[]const u8, registry: config.Models, name: []const u8, diag: *config.Diagnostic) !Resolved {
    if (registry.find(name)) |entry| {
        const kind = entry.kind orelse .generation;
        if (kind != .embedding) {
            diag.set("{s} is a registry entry of a {s} model, not an embedding model (an entry with \"kind\": \"embedding\")", .{ name, kindName(kind) });
            return error.NotAnEmbeddingModel;
        }
        const located = if (entry.path) |p| p else if (entry.repo != null and entry.file != null) try std.fs.path.join(arena, &.{ entry.repo.?, entry.file.? }) else {
            diag.set("models.{s} locates nothing: give it path, or repo and file", .{name});
            return error.InvalidRegistryEntry;
        };
        const path = try models(arena, root, located);
        const mmproj = if (entry.mmproj) |m| try std.fs.path.join(arena, &.{ std.fs.path.dirname(path) orelse ".", m }) else null;
        return .{ .path = path, .mmproj = mmproj };
    }
    if (catalog.findEmbedding(name)) |entry| {
        const dir = try models(arena, root, "");
        return .{
            .path = try std.fs.path.join(arena, &.{ dir, entry.repo, entry.file }),
            .mmproj = if (entry.mmproj) |m| try std.fs.path.join(arena, &.{ dir, m.repo, m.file }) else null,
            .entry = entry,
        };
    }
    const other: ?catalog.ModelKind = if (catalog.find(name) != null) .generation else if (catalog.findDecision(name) != null) .decision else null;
    if (other) |kind| {
        diag.set("{s} is a {s} model of the catalogue, not an embedding model ({s} is one)", .{ name, kindName(kind), catalog.embedding_entries[0].name });
        return error.NotAnEmbeddingModel;
    }
    return .{ .path = try models(arena, root, name) };
}

fn kindName(kind: catalog.ModelKind) []const u8 {
    return switch (kind) {
        .generation => "language",
        .decision => "decision",
        .embedding => "embedding",
    };
}

fn models(arena: std.mem.Allocator, root: ?[]const u8, path: []const u8) ![]const u8 {
    if (std.fs.path.isAbsolute(path) or std.mem.startsWith(u8, path, "./") or std.mem.startsWith(u8, path, "../")) return path;
    const dir = root orelse return error.MissingHome;
    return std.fs.path.join(arena, &.{ dir, "models", path });
}

pub const Located = struct {
    path: []const u8,
    /// The projector when the name has one and it is pulled.
    mmproj: ?[]const u8,
    /// `sha256` is the pull's (its sidecar), else empty: `digest` computes
    /// it for a file nuclis did not verify.
    identity: response.Identity,
};

/// `resolve`, then the file's presence (`ModelFileNotFound` names the pull
/// that fetches it) and its provenance from the pull's sidecar.
pub fn locate(arena: std.mem.Allocator, io: std.Io, root: ?[]const u8, registry: config.Models, name: []const u8, diag: *config.Diagnostic) !Located {
    const resolved = try resolve(arena, root, registry, name, diag);
    std.Io.Dir.cwd().access(io, resolved.path, .{}) catch {
        if (resolved.entry) |e|
            diag.set("{s}: not pulled yet (`nuclis model pull {s}` fetches it, {d} MB; --with mmproj adds images and audio)", .{ name, e.name, e.size / 1_000_000 })
        else
            diag.set("{s}: no file (an embedding registry entry, an embedding catalogue name, or a GGUF path)", .{resolved.path});
        return error.ModelFileNotFound;
    };
    var identity: response.Identity = .{ .name = name, .sha256 = "" };
    if (model.readSidecar(arena, io, .cwd(), try model.sidecarPath(arena, resolved.path)) catch null) |sidecar| {
        identity.repo = sidecar.repo;
        identity.revision = sidecar.revision;
        if (sidecar.sha256.len == 64) identity.sha256 = sidecar.sha256;
    }
    const mmproj = if (resolved.mmproj) |p| if (std.Io.Dir.cwd().access(io, p, .{})) |_| p else |_| null else null;
    return .{ .path = resolved.path, .mmproj = mmproj, .identity = identity };
}

/// The SHA-256 of a whole file as 64 lowercase hex digits, for a file
/// without a sidecar.
pub fn digest(arena: std.mem.Allocator, io: std.Io, path: []const u8) ![]const u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var buffer: [64 * 1024]u8 = undefined;
    var reader = file.reader(io, &buffer);
    var hash: std.crypto.hash.sha2.Sha256 = .init(.{});
    var chunk: [64 * 1024]u8 = undefined;
    while (true) {
        const n = try reader.interface.readSliceShort(&chunk);
        if (n == 0) break;
        hash.update(chunk[0..n]);
    }
    return arena.dupe(u8, &std.fmt.bytesToHex(hash.finalResult(), .lower));
}

test "names resolve: embedding entries, the catalogue, paths; other kinds refused" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diag: config.Diagnostic = .{};
    const registry: config.Models = .{ .entries = &.{
        .{ .name = "bf16", .entry = .{ .kind = .embedding, .repo = "unsloth/embeddinggemma-2-GGUF", .file = "embeddinggemma-2-BF16.gguf", .mmproj = "mmproj-BF16.gguf" } },
        .{ .name = "local", .entry = .{ .kind = .embedding, .path = "/data/e.gguf" } },
        .{ .name = "qwen", .entry = .{ .repo = "unsloth/Qwen3.8-27B-GGUF", .file = "Qwen3.8-27B-UD-Q4_K_M.gguf" } },
        .{ .name = "multi", .entry = .{ .kind = .decision, .path = "/data/laya" } },
    } };
    const pinned = try resolve(arena, "/r", registry, "embeddinggemma-2", &diag);
    try std.testing.expectEqualStrings("/r/models/unsloth/embeddinggemma-2-GGUF/embeddinggemma-2-Q8_0.gguf", pinned.path);
    try std.testing.expectEqualStrings("/r/models/unsloth/embeddinggemma-2-GGUF/mmproj-BF16.gguf", pinned.mmproj.?);
    try std.testing.expect(pinned.entry != null);
    const bf16 = try resolve(arena, "/r", registry, "bf16", &diag);
    try std.testing.expectEqualStrings("/r/models/unsloth/embeddinggemma-2-GGUF/embeddinggemma-2-BF16.gguf", bf16.path);
    try std.testing.expectEqualStrings("/r/models/unsloth/embeddinggemma-2-GGUF/mmproj-BF16.gguf", bf16.mmproj.?);
    try std.testing.expectEqualStrings("/data/e.gguf", (try resolve(arena, "/r", registry, "local", &diag)).path);
    try std.testing.expectEqualStrings("./e.gguf", (try resolve(arena, null, registry, "./e.gguf", &diag)).path);
    try std.testing.expectEqualStrings("/r/models/me/e.gguf", (try resolve(arena, "/r", registry, "me/e.gguf", &diag)).path);
    try std.testing.expectError(error.NotAnEmbeddingModel, resolve(arena, "/r", registry, "qwen", &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message(), "language") != null);
    try std.testing.expectError(error.NotAnEmbeddingModel, resolve(arena, "/r", registry, "multi", &diag));
    try std.testing.expectError(error.NotAnEmbeddingModel, resolve(arena, "/r", registry, "qwen3.8-27b", &diag));
    try std.testing.expectError(error.NotAnEmbeddingModel, resolve(arena, "/r", registry, "laya", &diag));
    try std.testing.expectError(error.MissingHome, resolve(arena, null, registry, "embeddinggemma-2", &diag));
}

test "the digest of a file is its SHA-256" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "f", .data = "abc" });
    const path = try tmp.dir.realPathFileAlloc(std.testing.io, "f", arena);
    try std.testing.expectEqualStrings("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", try digest(arena, std.testing.io, path));
}
