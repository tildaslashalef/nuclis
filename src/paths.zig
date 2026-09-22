//! User-directory policy belongs to the application, not the inference library.
//! Pass environment values in as data so resolution can be tested without changing
//! the process environment or touching the user's filesystem.
const std = @import("std");
const catalog = @import("catalog.zig");
const config = @import("config.zig");
const Allocator = std.mem.Allocator;

pub const config_file = "nuclis.json";

/// The user root: `NUCLIS_HOME` when set (absolute, non-empty), else
/// `$HOME/.nuclis`. Null when neither variable is set; commands that need
/// the root then fail with `MissingHome`, while an explicit `--model` still
/// works. Caller-owned storage.
pub fn root(alloc: Allocator, override: ?[]const u8, home: ?[]const u8) !?[]u8 {
    if (override) |dir| {
        if (dir.len == 0 or !std.fs.path.isAbsolute(dir)) return error.InvalidNuclisHome;
        return try alloc.dupe(u8, dir);
    }
    const dir = home orelse return null;
    if (dir.len == 0 or !std.fs.path.isAbsolute(dir)) return error.InvalidHome;
    return try std.fs.path.join(alloc, &.{ dir, ".nuclis" });
}

/// `<root>/nuclis.json`. Caller-owned storage.
pub fn configPath(alloc: Allocator, root_dir: []const u8) ![]u8 {
    return std.fs.path.join(alloc, &.{ root_dir, config_file });
}

/// The agent's data root, `<root>/agent`, optionally joined with a relative
/// path inside it (`"history.jsonl"`, `"sessions"`, `"exports"`; see
/// docs/spec.md § Sessions and storage). Nothing here creates a
/// directory: the caller decides when the agent's state is written, so a
/// command that never stores anything leaves the filesystem untouched.
/// Caller-owned storage.
pub fn agentPath(alloc: Allocator, root_dir: []const u8, inside: ?[]const u8) ![]u8 {
    const relative = inside orelse return std.fs.path.join(alloc, &.{ root_dir, agent_dir });
    if (relative.len == 0 or std.fs.path.isAbsolute(relative)) return error.InvalidAgentPath;
    return std.fs.path.join(alloc, &.{ root_dir, agent_dir, relative });
}

/// Named beside `config_file` so the layout is stated once.
pub const agent_dir = "agent";
pub const history_file = "history.jsonl";
pub const sessions_dir = "sessions";
pub const exports_dir = "exports";

/// The model file to open. `explicit` (a `--model` flag) or, without it,
/// `configured` (the file's `engine.model`) is tried in order as a
/// registry entry name (`registry`, the file's `models` map: its `path`
/// under `<root>/models` unless absolute, or `<root>/models/<repo>/<file>`),
/// as a catalogue name (`qwen3.8-27b` → the entry's file under
/// `<root>/models`), and last as a path: an explicit value is a path as
/// given, relative to the working directory like any command-line path
/// and needing no root, and a configured value resolves against
/// `<root>/models` unless it is absolute. Caller-owned storage.
pub fn modelPath(alloc: Allocator, explicit: ?[]const u8, configured: []const u8, root_dir: ?[]const u8, registry: config.Models) ![]u8 {
    const value = explicit orelse configured;
    if (value.len == 0) return error.EmptyModelPath;
    if (registry.find(value)) |entry| {
        if (entry.path) |path| {
            if (std.fs.path.isAbsolute(path)) return alloc.dupe(u8, path);
            const dir = root_dir orelse return error.MissingHome;
            return std.fs.path.join(alloc, &.{ dir, "models", path });
        }
        const dir = root_dir orelse return error.MissingHome;
        return std.fs.path.join(alloc, &.{ dir, "models", entry.repo.?, entry.file.? });
    }
    if (catalog.find(value)) |entry| {
        const dir = root_dir orelse return error.MissingHome;
        const models = try std.fs.path.join(alloc, &.{ dir, "models" });
        defer alloc.free(models);
        return catalog.localPath(alloc, models, entry, entry.file);
    }
    if (explicit != null or std.fs.path.isAbsolute(value)) return alloc.dupe(u8, value);
    const dir = root_dir orelse return error.MissingHome;
    return std.fs.path.join(alloc, &.{ dir, "models", value });
}

test "explicit model path takes precedence without needing a root" {
    const path = try modelPath(std.testing.allocator, "relative.gguf", "qwen/x.gguf", null, .{});
    defer std.testing.allocator.free(path);
    try std.testing.expectEqualStrings("relative.gguf", path);
    try std.testing.expectError(error.EmptyModelPath, modelPath(std.testing.allocator, "", "qwen/x.gguf", "/r", .{}));
    try std.testing.expectError(error.MissingHome, modelPath(std.testing.allocator, null, "qwen/x.gguf", null, .{}));
}

test "resolve the root from HOME or the override" {
    const alloc = std.testing.allocator;
    const normal = (try root(alloc, null, "/users/example")).?;
    defer alloc.free(normal);
    try std.testing.expectEqualStrings("/users/example/.nuclis", normal);
    const alternate = (try root(alloc, "/tmp/nuclis", "/users/example")).?;
    defer alloc.free(alternate);
    try std.testing.expectEqualStrings("/tmp/nuclis", alternate);
    try std.testing.expect((try root(alloc, null, null)) == null);
    try std.testing.expectError(error.InvalidNuclisHome, root(alloc, "", null));
    try std.testing.expectError(error.InvalidNuclisHome, root(alloc, "relative", null));
    try std.testing.expectError(error.InvalidHome, root(alloc, null, "relative"));
    const settings_path = try configPath(alloc, normal);
    defer alloc.free(settings_path);
    try std.testing.expectEqualStrings("/users/example/.nuclis/nuclis.json", settings_path);
}

test "the agent's data root is <root>/agent, with its files inside it" {
    const alloc = std.testing.allocator;
    const dir = try agentPath(alloc, "/users/example/.nuclis", null);
    defer alloc.free(dir);
    try std.testing.expectEqualStrings("/users/example/.nuclis/agent", dir);
    const history = try agentPath(alloc, "/users/example/.nuclis", history_file);
    defer alloc.free(history);
    try std.testing.expectEqualStrings("/users/example/.nuclis/agent/history.jsonl", history);
    const sessions = try agentPath(alloc, "/users/example/.nuclis", sessions_dir);
    defer alloc.free(sessions);
    try std.testing.expectEqualStrings("/users/example/.nuclis/agent/sessions", sessions);
    // The root is the only place the agent writes: an absolute or empty
    // inner path is a caller mistake, not a path outside the root.
    try std.testing.expectError(error.InvalidAgentPath, agentPath(alloc, "/r", "/etc/passwd"));
    try std.testing.expectError(error.InvalidAgentPath, agentPath(alloc, "/r", ""));
}

test "catalogue names resolve to the entry's file under <root>/models, flagged or configured" {
    const alloc = std.testing.allocator;
    const flagged = try modelPath(alloc, "qwen3.8-27b", "other.gguf", "/tmp/nuclis", .{});
    defer alloc.free(flagged);
    try std.testing.expectEqualStrings("/tmp/nuclis/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf", flagged);
    const configured = try modelPath(alloc, null, "qwen3.8-27b", "/tmp/nuclis", .{});
    defer alloc.free(configured);
    try std.testing.expectEqualStrings(flagged, configured);
    try std.testing.expectError(error.MissingHome, modelPath(alloc, "qwen3.8-27b", "x", null, .{}));
    try std.testing.expectError(error.MissingHome, modelPath(alloc, null, "qwen3.8-27b", null, .{}));
}

test "configured model names resolve under <root>/models unless absolute" {
    const alloc = std.testing.allocator;
    const relative = try modelPath(alloc, null, "qwen/x.gguf", "/tmp/nuclis", .{});
    defer alloc.free(relative);
    try std.testing.expectEqualStrings("/tmp/nuclis/models/qwen/x.gguf", relative);
    const absolute = try modelPath(alloc, null, "/models/y.gguf", "/tmp/nuclis", .{});
    defer alloc.free(absolute);
    try std.testing.expectEqualStrings("/models/y.gguf", absolute);
    try std.testing.expectError(error.EmptyModelPath, modelPath(alloc, null, "", "/tmp/nuclis", .{}));
}

test "registry entries resolve before the catalogue and before paths" {
    const alloc = std.testing.allocator;
    const entries = [_]config.NamedModel{
        .{ .name = "gemma", .entry = .{ .repo = "unsloth/gemma-4-12b-it-GGUF", .file = "g.gguf" } },
        .{ .name = "rel", .entry = .{ .path = "custom/m.gguf" } },
        .{ .name = "abs", .entry = .{ .path = "/scratch/m.gguf" } },
        // Shadows the catalogue name on purpose: the file wins.
        .{ .name = "qwen3.8-27b", .entry = .{ .path = "mine/q.gguf" } },
    };
    const registry: config.Models = .{ .entries = &entries };
    const by_repo = try modelPath(alloc, null, "gemma", "/tmp/nuclis", registry);
    defer alloc.free(by_repo);
    try std.testing.expectEqualStrings("/tmp/nuclis/models/unsloth/gemma-4-12b-it-GGUF/g.gguf", by_repo);
    const relative = try modelPath(alloc, "rel", "x", "/tmp/nuclis", registry);
    defer alloc.free(relative);
    try std.testing.expectEqualStrings("/tmp/nuclis/models/custom/m.gguf", relative);
    const absolute = try modelPath(alloc, null, "abs", null, registry);
    defer alloc.free(absolute);
    try std.testing.expectEqualStrings("/scratch/m.gguf", absolute);
    const shadowed = try modelPath(alloc, null, "qwen3.8-27b", "/tmp/nuclis", registry);
    defer alloc.free(shadowed);
    try std.testing.expectEqualStrings("/tmp/nuclis/models/mine/q.gguf", shadowed);
    try std.testing.expectError(error.MissingHome, modelPath(alloc, "gemma", "x", null, registry));
    try std.testing.expectError(error.MissingHome, modelPath(alloc, "rel", "x", null, registry));
    // A name the registry does not know falls through to the catalogue, then to a path.
    const fallthrough = try modelPath(alloc, "other.gguf", "x", "/tmp/nuclis", registry);
    defer alloc.free(fallthrough);
    try std.testing.expectEqualStrings("other.gguf", fallthrough);
}
