//! Decision models by name: resolve a name to a checkpoint directory (the
//! registry first, then the decision catalogue, then a path), find what is
//! pulled, and list every decision model with its budget. Shared by `nuclis
//! decide` and the API; the text catalogue's names are refused.
const std = @import("std");
const inference = @import("inference");
const config = @import("../config.zig");
const catalog = @import("../catalog.zig");
const model = @import("../model.zig");
const response = @import("response.zig");

const profile = inference.profiles.laya;

/// The checkpoint directory `name` names, in order: a registry entry (which
/// must be of kind `decision`), a decision catalogue name, else a directory
/// (as given when absolute or `./`, else under `<root>/models`). A text
/// model's name is refused.
pub fn resolve(arena: std.mem.Allocator, root: ?[]const u8, registry: config.Models, name: []const u8, diag: *config.Diagnostic) ![]const u8 {
    if (registry.find(name)) |entry| {
        if (entry.kind != .decision) {
            diag.set("{s} is a registry entry of a text model; `nuclis decide` opens a decision checkpoint (an entry with \"kind\": \"decision\")", .{name});
            return error.NotADecisionModel;
        }
        const located = if (entry.path) |p| p else if (entry.repo != null and entry.file != null) try std.fs.path.join(arena, &.{ entry.repo.?, entry.file.? }) else {
            diag.set("models.{s} locates nothing: give it path, or repo and file", .{name});
            return error.InvalidRegistryEntry;
        };
        const full = try models(arena, root, located);
        // A path to the weights names their directory.
        return if (std.mem.endsWith(u8, full, ".safetensors")) std.fs.path.dirname(full) orelse full else full;
    }
    if (catalog.findDecision(name)) |entry| return catalog.decisionDirectory(arena, try models(arena, root, ""), entry);
    if (catalog.find(name) != null) {
        diag.set("{s} is a text model of the catalogue; `nuclis decide` opens a decision checkpoint (laya)", .{name});
        return error.NotADecisionModel;
    }
    return models(arena, root, name);
}

fn models(arena: std.mem.Allocator, root: ?[]const u8, path: []const u8) ![]const u8 {
    if (std.fs.path.isAbsolute(path) or std.mem.startsWith(u8, path, "./") or std.mem.startsWith(u8, path, "../")) return path;
    const dir = root orelse return error.MissingHome;
    return std.fs.path.join(arena, &.{ dir, "models", path });
}

pub const Located = struct {
    directory: []const u8,
    identity: response.Identity,
    /// clef's backbone GGUF when the catalogue entry names one.
    backbone: ?[]const u8 = null,
    /// clef's projector when the catalogue entry names one and it is pulled.
    mmproj: ?[]const u8 = null,

    pub fn location(self: Located) inference.decide.Location {
        return .{ .directory = self.directory, .backbone = self.backbone, .mmproj = self.mmproj };
    }
};

/// `resolve`, then the weights' presence (`ModelFileNotFound` names the
/// pull that fetches them) and their provenance from the pull's sidecar.
pub fn locate(arena: std.mem.Allocator, io: std.Io, root: ?[]const u8, registry: config.Models, name: []const u8, diag: *config.Diagnostic) !Located {
    const directory = try resolve(arena, root, registry, name, diag);
    const entry = if (registry.find(name) == null) catalog.findDecision(name) else null;
    const weights_name = if (entry) |e| std.fs.path.basename(e.file) else if (inference.decide.Family.of(io, directory) == .clef) inference.models.clef.head_file else "model.safetensors";
    const weights = try std.fs.path.join(arena, &.{ directory, weights_name });
    const backbone = if (entry) |e| if (e.backbone) |b| try std.fs.path.join(arena, &.{ try models(arena, root, ""), b.repo, b.file }) else null else null;
    for ([_]?[]const u8{ weights, backbone }) |path| if (path) |p| std.Io.Dir.cwd().access(io, p, .{}) catch {
        if (entry) |e|
            diag.set("{s}: not pulled yet (`nuclis model pull {s}` fetches it, {d} MB)", .{ name, e.name, e.totalSize() / 1_000_000 })
        else
            diag.set("{s}: no model.safetensors (a Laya checkpoint directory, a decision registry entry, or a decision catalogue name)", .{directory});
        return error.ModelFileNotFound;
    };
    var identity: response.Identity = .{ .name = name };
    if (model.readSidecar(arena, io, .cwd(), try model.sidecarPath(arena, weights)) catch null) |sidecar| {
        identity.repo = sidecar.repo;
        identity.revision = sidecar.revision;
    }
    // The projector is optional: without it a request with images is refused.
    const mmproj = if (entry) |e| if (e.mmproj) |m| try std.fs.path.join(arena, &.{ try models(arena, root, ""), m.repo, m.file }) else null else null;
    const pulled_mmproj = if (mmproj) |p| if (std.Io.Dir.cwd().access(io, p, .{})) |_| p else |_| null else null;
    return .{ .directory = directory, .identity = identity, .backbone = backbone, .mmproj = pulled_mmproj };
}

/// A decision model of the catalogue or the registry, as it stands on disk.
pub const Listed = struct {
    name: []const u8,
    /// The catalogue's or the registry's, else the pull's.
    repo: ?[]const u8,
    revision: ?[]const u8,
    /// Null when the name resolves to nothing (no root, a bad entry).
    directory: ?[]const u8,
    present: bool,
    /// From `rl_agent_config.json` when it is present and valid.
    budget: ?profile.Budget,
    /// From the catalogue entry (a backbone means clef), else the files.
    family: inference.decide.Family = .laya,
    /// clef with its projector pulled.
    images: bool = false,
};

/// Every decision model: the catalogue's, then the registry's decision
/// entries; a registry entry shadows the catalogue name it reuses, as in
/// `resolve`. `arena` owns everything.
pub fn list(arena: std.mem.Allocator, io: std.Io, root: ?[]const u8, registry: config.Models) ![]Listed {
    var out: std.ArrayList(Listed) = .empty;
    for (&catalog.decision_entries) |*entry| {
        if (registry.find(entry.name) != null) continue;
        try out.append(arena, try describe(arena, io, root, registry, entry.name, entry.repo, entry.revision));
    }
    for (registry.entries) |named| {
        if (named.entry.kind != .decision) continue;
        try out.append(arena, try describe(arena, io, root, registry, named.name, named.entry.repo, named.entry.revision));
    }
    return out.items;
}

fn describe(arena: std.mem.Allocator, io: std.Io, root: ?[]const u8, registry: config.Models, name: []const u8, repo: ?[]const u8, revision: ?[]const u8) !Listed {
    var listed: Listed = .{ .name = name, .repo = repo, .revision = revision, .directory = null, .present = false, .budget = null };
    // Known before anything is pulled, so a client plans for it either way.
    if (registry.find(name) == null) if (catalog.findDecision(name)) |e| if (e.backbone != null) {
        listed.family = .clef;
    };
    var diag: config.Diagnostic = .{};
    const located = locate(arena, io, root, registry, name, &diag) catch |err| switch (err) {
        error.OutOfMemory => return err,
        error.ModelFileNotFound => {
            listed.directory = resolve(arena, root, registry, name, &diag) catch null;
            return listed;
        },
        else => return listed,
    };
    listed.directory = located.directory;
    listed.present = true;
    if (listed.family == .laya) listed.family = inference.decide.Family.of(io, located.directory);
    listed.images = listed.family == .clef and located.mmproj != null;
    if (located.identity.repo) |r| listed.repo = r;
    if (located.identity.revision) |r| listed.revision = r;
    const agent_path = try std.fs.path.join(arena, &.{ located.directory, "rl_agent_config.json" });
    if (std.Io.Dir.cwd().readFileAlloc(io, agent_path, arena, .limited(1024 * 1024))) |bytes| {
        if (profile.parseAgentConfig(arena, bytes)) |agent| listed.budget = agent.budget else |_| {}
    } else |_| {}
    return listed;
}

test "model names resolve: decision entries, the catalogue, paths; text models refused" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diag: config.Diagnostic = .{};
    const registry: config.Models = .{ .entries = &.{
        .{ .name = "multi", .entry = .{ .kind = .decision, .repo = "convaiinnovations/laya", .file = "multilingual/model.safetensors" } },
        .{ .name = "local", .entry = .{ .kind = .decision, .path = "/data/laya" } },
        .{ .name = "qwen", .entry = .{ .repo = "unsloth/Qwen3.8-27B-GGUF", .file = "Qwen3.8-27B-UD-Q4_K_M.gguf" } },
    } };
    try std.testing.expectEqualStrings("/r/models/convaiinnovations/laya", try resolve(arena, "/r", registry, "laya", &diag));
    try std.testing.expectEqualStrings("/r/models/convaiinnovations/laya/multilingual", try resolve(arena, "/r", registry, "multi", &diag));
    try std.testing.expectEqualStrings("/data/laya", try resolve(arena, "/r", registry, "local", &diag));
    try std.testing.expectEqualStrings("/r/models/me/finetune", try resolve(arena, "/r", registry, "me/finetune", &diag));
    try std.testing.expectEqualStrings("./here", try resolve(arena, null, registry, "./here", &diag));
    try std.testing.expectError(error.NotADecisionModel, resolve(arena, "/r", registry, "qwen", &diag));
    try std.testing.expectError(error.NotADecisionModel, resolve(arena, "/r", registry, "qwen3.8-27b", &diag));
    try std.testing.expectError(error.MissingHome, resolve(arena, null, registry, "laya", &diag));
}

test "the listing: catalogue then registry, nothing pulled under an empty root" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(std.testing.io, ".", arena);
    const registry: config.Models = .{ .entries = &.{
        .{ .name = "laya", .entry = .{ .kind = .decision, .path = "/nowhere/laya" } },
        .{ .name = "qwen", .entry = .{ .repo = "unsloth/Qwen3.8-27B-GGUF", .file = "q.gguf" } },
    } };
    const listed = try list(arena, std.testing.io, root, registry);
    try std.testing.expectEqual(@as(usize, catalog.decision_entries.len), listed.len);
    try std.testing.expectEqualStrings("laya-multilingual", listed[0].name);
    try std.testing.expectEqualStrings("laya", listed[listed.len - 1].name);
    try std.testing.expectEqualStrings("/nowhere/laya", listed[listed.len - 1].directory.?);
    for (listed) |l| try std.testing.expect(!l.present and l.budget == null and !l.images);
    for (listed) |l| try std.testing.expectEqual(std.mem.eql(u8, l.name, "clef-flash"), l.family == .clef);
}
