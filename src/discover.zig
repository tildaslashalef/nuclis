//! `nuclis config init` and `nuclis config sync`: registry entries for what
//! is installed under `<root>/models`, and the entries whose file is gone.
//!
//! An installed index artifact of the chat kind gets the entry
//! `config.registryEntry` writes (its pinned files and measured defaults);
//! other kinds resolve by their index name and need none, and `legacy`
//! artifacts are never written. A file outside the index is judged as
//! `model inspect` judges a directory (an adapter for its architecture,
//! every tensor in the executable set, the binding) or, for safetensors, by
//! the files beside it; its profile is chosen by template digest and forced
//! to the family's when no profile matches (a finetune). Decisions are pure
//! over the listing; `config.registerEntries` writes the file.
const std = @import("std");
const inference = @import("inference");
const config = @import("config.zig");
const catalog = @import("catalog.zig");
const model = @import("model.zig");
const style = @import("tui/style.zig");
const Allocator = std.mem.Allocator;

pub const Candidate = struct {
    name: []const u8,
    /// Relative to the models directory: `<owner>/<repo>/<file>`.
    path: []const u8,
    kind: catalog.ModelKind,
    /// The index artifact, null for a file outside the index.
    artifact: ?[]const u8 = null,
    architecture: ?[]const u8 = null,
    adapter: ?[]const u8 = null,
    /// The template digest matched no profile, so the family's is forced.
    profile_forced: bool = false,
    entry: config.ModelEntry,
};

pub const Skipped = struct { path: []const u8, reason: []const u8 };

pub const Report = struct {
    schema_version: u32 = 2,
    models_dir: []const u8,
    config_file: []const u8 = "",
    dry_run: bool = false,
    written: bool = false,
    /// `--prune`: `gone` is removed from the file (with `dry_run`, would be).
    pruned: bool = false,
    /// `engine.model` when it names nothing installed.
    model_absent: ?[]const u8 = null,
    /// Whether the default `engine.model` is installed.
    default_installed: bool = false,
    registered: []const Candidate,
    /// Registry entries whose file is not on disk.
    gone: []const model.MissingEntry,
    skipped: []const Skipped,

    pub fn render(self: Report, out: *std.Io.Writer, json: bool, sty: style.Style) !void {
        if (json) {
            try std.json.Stringify.value(self, .{ .whitespace = .indent_2, .emit_null_optional_fields = false }, out);
            return out.writeByte('\n');
        }
        const off = sty.off();
        if (self.registered.len == 0) {
            try out.print("{s}nothing to register{s} under {s}{s}{s}", .{ sty.on(.bold), off, sty.on(.code), self.models_dir, off });
        } else if (self.dry_run) {
            try out.print("{s}dry run:{s} {d} model(s) under {s}{s}{s} would be registered in {s}{s}{s}; nothing written", .{ sty.on(.bold), off, self.registered.len, sty.on(.code), self.models_dir, off, sty.on(.code), self.config_file, off });
        } else {
            try out.print("{s}registered{s} {d} model(s) under {s}{s}{s} in {s}{s}{s}", .{ sty.on(.success), off, self.registered.len, sty.on(.code), self.models_dir, off, sty.on(.code), self.config_file, off });
        }
        try out.writeByte('\n');
        try self.renderRows(out, sty);
    }

    /// The candidates, the entries whose file is gone, and what was skipped,
    /// without the summary line (`config init` says what it wrote first).
    pub fn renderRows(self: Report, out: *std.Io.Writer, sty: style.Style) !void {
        const off = sty.off();
        for (self.registered) |c| {
            try out.print("  {s}{s}{s}  {s}{s}{s}\n", .{ sty.on(.keyword), c.name, off, sty.on(.code), c.path, off });
            try out.print("      {s}{s}", .{ sty.on(.dim), catalog.groupName(c.kind) });
            if (c.artifact != null) try out.writeAll(", from the index");
            if (c.adapter) |a| try out.print(", {s} adapter", .{a});
            if (c.entry.profile) |p| try out.print(", profile {s}{s}", .{ @tagName(p), if (c.profile_forced) " (forced: the template is not a pinned one)" else "" });
            if (c.entry.mmproj) |m| try out.print(", mmproj {s}", .{m});
            if (c.entry.draft) |m| try out.print(", draft {s}", .{m});
            if (c.entry.generation.speculative) |s| try out.print(", speculative {s}", .{if (s) "on" else "off"});
            try out.print("{s}\n", .{off});
        }
        if (self.gone.len > 0) {
            if (self.pruned) {
                try out.print("{s}{s}{s} {d} entr{s} whose file is gone\n", .{ sty.on(.success), if (self.dry_run) "would remove" else "removed", off, self.gone.len, if (self.gone.len == 1) "y" else "ies" });
            } else try out.print("{s}entries whose file is gone{s} {s}(`nuclis config sync --prune` removes them){s}\n", .{ sty.on(.header), off, sty.on(.dim), off });
            for (self.gone) |g| try out.print("  {s}{s}{s}  {s}{s}{s}\n", .{ sty.on(.keyword), g.name, off, sty.on(.code), g.path, off });
        }
        if (self.skipped.len > 0) {
            try out.print("{s}skipped{s}\n", .{ sty.on(.header), off });
            for (self.skipped) |s| try out.print("  {s}{s}{s}  {s}{s}{s}\n", .{ sty.on(.code), s.path, off, sty.on(.dim), s.reason, off });
        }
        if (self.model_absent) |m| try out.print("{s}notice:{s} engine.model {s}{s}{s} is not installed {s}(`nuclis model pull {s}`, or `nuclis config set engine.model <name>`){s}\n", .{ sty.on(.warning), off, sty.on(.keyword), m, off, sty.on(.dim), m, off });
    }
};

/// The installed models judged one by one: every decision lands in the
/// report as a candidate or a skipped row with its reason, and every
/// registry entry without a file as `gone`.
/// `engine_model`, when given, is reported when nothing installed answers to it.
pub fn plan(arena: Allocator, gpa: Allocator, io: std.Io, root: []const u8, registry: config.Models, engine_model: ?[]const u8) !Report {
    const listing = try model.list(arena, io, root, registry, null);
    var registered: std.ArrayList(Candidate) = .empty;
    var skipped: std.ArrayList(Skipped) = .empty;
    var taken: std.ArrayList([]const u8) = .empty;
    for (registry.entries) |e| try taken.append(arena, e.name);
    for (&catalog.artifacts) |*a| try taken.append(arena, a.name);
    for (listing.installed) |row| {
        const main = row.files[0];
        if (main.role != .main) continue;
        if (row.artifact) |name| {
            const a = catalog.find(name).?;
            if (row.registered != null or registry.find(name) != null) continue;
            if (a.kind != .generation or a.status == .legacy or main.status == .mismatch) continue;
            try registered.append(arena, .{ .name = name, .path = main.path, .kind = .generation, .artifact = name, .architecture = a.architecture, .entry = config.registryEntry(a) });
            continue;
        }
        if (row.registered != null) continue;
        const candidate = try judgeRow(arena, gpa, io, listing.models_dir, row, &skipped) orelse continue;
        var c = candidate;
        c.name = try uniqueName(arena, main.path, taken.items);
        try taken.append(arena, c.name);
        try registered.append(arena, c);
    }
    std.mem.sort(Candidate, registered.items, {}, struct {
        fn lessThan(_: void, a: Candidate, b: Candidate) bool {
            return std.mem.lessThan(u8, a.name, b.name);
        }
    }.lessThan);
    return .{ .models_dir = listing.models_dir, .registered = registered.items, .gone = listing.missing, .skipped = skipped.items, .default_installed = listing.find((config.Config.Engine{}).model) != null, .model_absent = if (engine_model) |m| if (listing.find(m) == null) m else null else null };
}

/// A file outside the index as a candidate, or null with the reason in
/// `skipped`. The entry locates it the way `model pull --register` would
/// when its sidecar names the pull (so `model pull <name>` works on it),
/// else by `path`.
fn judgeRow(arena: Allocator, gpa: Allocator, io: std.Io, models: []const u8, row: model.Installed, skipped: *std.ArrayList(Skipped)) !?Candidate {
    const main = row.files[0];
    var entry: config.ModelEntry = .{ .path = main.path };
    if (sidecarOf(arena, io, models, main.path)) |s| if (s.revision != null and std.mem.eql(u8, main.path, try std.fs.path.join(arena, &.{ s.repo, s.file }))) {
        entry = .{ .repo = s.repo, .file = s.file, .revision = s.revision };
    };
    const kind = row.kind orelse {
        try skipped.append(arena, .{ .path = main.path, .reason = if (row.architecture) |a| try arena.print("no adapter for architecture \"{s}\"", .{a}) else "nothing says what runs it (no adapter for its header, or a safetensors layout nuclis does not open)" });
        return null;
    };
    for (row.files[1..]) |f| switch (f.role) {
        .mmproj => entry.mmproj = relativeTo(main.path, f.path),
        .draft => entry.draft = relativeTo(main.path, f.path),
        else => {},
    };
    if (kind != .generation) {
        entry.kind = kind;
        // The weights of a safetensors checkpoint locate its directory.
        return .{ .name = "", .path = main.path, .kind = kind, .architecture = row.architecture, .entry = entry };
    }
    const absolute = try std.fs.path.join(arena, &.{ models, main.path });
    var doc = inference.gguf.open(gpa, io, absolute, .{}) catch |err| {
        try skipped.append(arena, .{ .path = main.path, .reason = try arena.print("the GGUF directory does not parse: {s}", .{@errorName(err)}) });
        return null;
    };
    defer doc.deinit();
    const architecture = try arena.dupe(u8, doc.string("general.architecture") orelse "");
    const adapter = switch (try judge(arena, gpa, &doc, architecture)) {
        .adapter => |a| a,
        .rejected => |reason| {
            try skipped.append(arena, .{ .path = main.path, .reason = reason });
            return null;
        },
    };
    const family = familyOf(architecture, blockCount(&doc, architecture)) orelse {
        try skipped.append(arena, .{ .path = main.path, .reason = try arena.print("no index family for architecture \"{s}\"", .{architecture}) });
        return null;
    };
    const pinned = inference.profiles.forDocument(doc);
    if (pinned == null) entry.profile = family.defaults.profile;
    // The family's measured verdict, except that a family whose drafter
    // is a companion file has nothing to draft with when it is absent.
    const verdict = family.verdict();
    entry.generation.speculative = verdict.speculative and (entry.draft != null or family.draft(null) == null);
    entry.generation.draft_length = verdict.draft_length;
    return .{ .name = "", .path = main.path, .kind = .generation, .architecture = architecture, .adapter = @tagName(adapter), .profile_forced = pinned == null, .entry = entry };
}

fn sidecarOf(arena: Allocator, io: std.Io, models: []const u8, path: []const u8) ?model.Sidecar {
    const absolute = std.fs.path.join(arena, &.{ models, path }) catch return null;
    const sidecar = model.sidecarPath(arena, absolute) catch return null;
    return model.readSidecar(arena, io, .cwd(), sidecar) catch null;
}

/// `file` relative to the directory of `main` (both relative to the models
/// directory; the listing puts companions in it or below it).
fn relativeTo(main: []const u8, file: []const u8) []const u8 {
    const dir = std.fs.path.dirname(main) orelse "";
    return if (dir.len > 0 and std.mem.startsWith(u8, file, dir) and file.len > dir.len) file[dir.len + 1 ..] else file;
}

const Judgement = union(enum) { adapter: inference.models.Adapter, rejected: []const u8 };

/// `model inspect`'s three checks on a local directory: an adapter for the
/// architecture, every tensor in its executable set, the binding.
fn judge(arena: Allocator, gpa: Allocator, doc: *const inference.gguf.Document, architecture: []const u8) !Judgement {
    const adapter = inference.models.adapterFor(architecture) orelse return .{ .rejected = if (architecture.len == 0)
        "the file declares no general.architecture"
    else
        try arena.print("no adapter for architecture \"{s}\"", .{architecture}) };
    for (doc.tensors) |tensor| if (!inference.models.registry.executableEncoding(adapter, tensor.encoding_id)) {
        const layout = inference.encoding.layout(tensor.encoding_id);
        return .{ .rejected = try arena.print("tensor {s} uses encoding {s} (id {d}), outside the {s} adapter's executable set", .{ tensor.name, if (layout) |l| l.name else "?", tensor.encoding_id, @tagName(adapter) }) };
    };
    _ = inference.models.registry.validate(adapter, gpa, doc) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return .{ .rejected = try arena.print("the {s} adapter rejects the file: {s}", .{ @tagName(adapter), @errorName(err) }) },
    };
    return .{ .adapter = adapter };
}

/// The index artifact that carries the family's profile and verdict: the
/// one whose `block_count` matches the file's, else the architecture's
/// first chat artifact.
pub fn familyOf(architecture: []const u8, blocks: ?u64) ?*const catalog.Artifact {
    var first: ?*const catalog.Artifact = null;
    for (&catalog.artifacts) |*a| if (a.kind == .generation and std.mem.eql(u8, a.architecture, architecture)) {
        if (blocks != null and a.traits.block_count == blocks) return a;
        if (first == null) first = a;
    };
    return first;
}

/// The file's `<architecture>.block_count`, when it states one.
fn blockCount(doc: *const inference.gguf.Document, architecture: []const u8) ?u64 {
    var key: [96]u8 = undefined;
    const name = std.fmt.bufPrint(&key, "{s}.block_count", .{architecture}) catch return null;
    return switch (doc.get(name) orelse return null) {
        .unsigned => |n| n,
        .signed => |n| if (n < 0) null else @intCast(n),
        else => null,
    };
}

/// The repository's name as an entry name: lower-case, `-gguf` dropped,
/// `[a-z0-9._-]` kept (other bytes become `-`); a taken name gains the
/// file's quantization suffix, then a counter. Never an index name.
pub fn uniqueName(arena: Allocator, path: []const u8, taken: []const []const u8) ![]const u8 {
    var parts = std.mem.splitScalar(u8, path, '/');
    _ = parts.next(); // owner
    const repo = parts.next() orelse path;
    const base = try sanitize(arena, repo);
    if (!isTaken(base, taken)) return base;
    const stem = std.fs.path.stem(std.fs.path.basename(path));
    const quant = if (std.mem.lastIndexOfScalar(u8, stem, '-')) |i| stem[i + 1 ..] else stem;
    const with_quant = try arena.print("{s}-{s}", .{ base, try sanitize(arena, quant) });
    if (!isTaken(with_quant, taken)) return with_quant;
    var n: usize = 2;
    while (n < 100) : (n += 1) {
        const numbered = try arena.print("{s}-{d}", .{ with_quant, n });
        if (!isTaken(numbered, taken)) return numbered;
    }
    return error.NoFreeName;
}

fn isTaken(name: []const u8, taken: []const []const u8) bool {
    for (taken) |t| if (std.mem.eql(u8, t, name)) return true;
    return false;
}

fn sanitize(arena: Allocator, text: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (text) |c| {
        const l = std.ascii.toLower(c);
        const keep = std.ascii.isAlphanumeric(l) or l == '.' or l == '_' or l == '-';
        if (keep) try out.append(arena, l) else if (out.items.len > 0 and out.items[out.items.len - 1] != '-') try out.append(arena, '-');
    }
    var name = std.mem.trim(u8, out.items, "-");
    if (std.mem.endsWith(u8, name, "-gguf")) name = name[0 .. name.len - 5];
    if (std.mem.endsWith(u8, name, ".gguf")) name = name[0 .. name.len - 5];
    if (name.len == 0) return error.NoFreeName;
    return name[0..@min(name.len, 64)];
}

/// The candidates as the registry entries `config.registerEntries` writes.
pub fn toEntries(arena: Allocator, candidates: []const Candidate) ![]const config.NamedModel {
    const out = try arena.alloc(config.NamedModel, candidates.len);
    for (candidates, out) |c, *d| d.* = .{ .name = c.name, .entry = c.entry };
    return out;
}

/// `engine.model` for a fresh file: the default when it is installed,
/// else the first installed chat model of the plan, else the default.
pub fn defaultModel(candidates: []const Candidate, installed_default: bool) []const u8 {
    const default = (config.Config.Engine{}).model;
    if (installed_default) return default;
    for (candidates) |c| if (c.kind == .generation) return c.name;
    return default;
}

// ----- tests -----

const testing = std.testing;

test "a file takes the verdict of the index artifact its layer count matches" {
    // Gemma 4's three checkpoints share an architecture and differ in verdict.
    const e4b = familyOf("gemma4", 42).?;
    try testing.expectEqualStrings("gemma-4-e4b-qat", e4b.name);
    try testing.expect(e4b.verdict().speculative and e4b.verdict().draft_length == 6);
    const moe = familyOf("gemma4", 30).?;
    try testing.expectEqualStrings("gemma-4-26b-a4b", moe.name);
    try testing.expect(!moe.verdict().speculative);
    try testing.expectEqualStrings("gemma-4-12b-qat", familyOf("gemma4", 48).?.name);
    // An unknown layer count, or none, falls back to the architecture's first artifact.
    try testing.expectEqualStrings("gemma-4-12b-qat", familyOf("gemma4", 7).?.name);
    try testing.expectEqualStrings("gemma-4-12b-qat", familyOf("gemma4", null).?.name);
    try testing.expectEqualStrings("qwen3.8-27b", familyOf("qwen35", 64).?.name);
    try testing.expect(familyOf("llama", 32) == null);
    // clef's backbone shares Qwen's architecture but is not a chat artifact.
    try testing.expectEqualStrings("qwen3.8-27b", familyOf("qwen35", null).?.name);
}

test "a companion is named relative to its main file's directory" {
    try testing.expectEqualStrings("mmproj-BF16.gguf", relativeTo("o/r/Main-Q4_K_M.gguf", "o/r/mmproj-BF16.gguf"));
    try testing.expectEqualStrings("MTP/mtp-main.gguf", relativeTo("o/r/Main-Q4_K_M.gguf", "o/r/MTP/mtp-main.gguf"));
}

test "names come from the repository, never collide, and never take an index name" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try testing.expectEqualStrings("gemma4-12b-qat-uncensored-hauhaucs-balanced", try uniqueName(arena, "HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf", &.{}));
    try testing.expectEqualStrings("ternary-bonsai-2-27b", try uniqueName(arena, "prism-ml/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf", &.{}));
    try testing.expectEqualStrings("ternary-bonsai-2-27b-pq2_0", try uniqueName(arena, "prism-ml/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf", &.{"ternary-bonsai-2-27b"}));
    try testing.expectEqualStrings("ternary-bonsai-2-27b-pq2_0-2", try uniqueName(arena, "prism-ml/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf", &.{ "ternary-bonsai-2-27b", "ternary-bonsai-2-27b-pq2_0" }));
    try testing.expectEqualStrings("a-b_c", try uniqueName(arena, "o/A  b_c.GGUF/x.gguf", &.{}));
}

test "the default model: the default when installed, else the first installed chat model" {
    const chat: Candidate = .{ .name = "gemma-4-e4b-qat", .path = "u/r/f.gguf", .kind = .generation, .entry = .{} };
    const speech: Candidate = .{ .name = "drama-3", .path = "s/d/f.gguf", .kind = .speech, .entry = .{} };
    try testing.expectEqualStrings("gemma-4-26b-a4b", defaultModel(&.{ speech, chat }, true));
    try testing.expectEqualStrings("gemma-4-e4b-qat", defaultModel(&.{ speech, chat }, false));
    try testing.expectEqualStrings("gemma-4-26b-a4b", defaultModel(&.{speech}, false));
}

test "a report renders both ways from the same rows" {
    const report: Report = .{
        .models_dir = "/m",
        .config_file = "/c.json",
        .registered = &.{.{ .name = "fine", .path = "o/r/f.gguf", .kind = .generation, .architecture = "gemma4", .adapter = "gemma4", .profile_forced = true, .entry = .{ .repo = "o/r", .file = "f.gguf", .revision = "abc", .mmproj = "mmproj.gguf", .profile = .gemma4, .generation = .{ .speculative = false, .draft_length = 4 } } }},
        .gone = &.{.{ .name = "old", .path = "o/gone/x.gguf" }},
        .skipped = &.{.{ .path = "o/r/mmproj.gguf", .reason = "a mmproj companion" }},
    };
    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    try report.render(&out.writer, false, .none);
    try testing.expect(std.mem.indexOf(u8, out.written(), "registered 1 model(s)") != null);
    try testing.expect(std.mem.indexOf(u8, out.written(), "profile gemma4 (forced") != null);
    try testing.expect(std.mem.indexOf(u8, out.written(), "config sync --prune") != null);
    out.clearRetainingCapacity();
    try report.render(&out.writer, true, .none);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, out.written(), .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("fine", parsed.value.object.get("registered").?.array.items[0].object.get("name").?.string);
    try testing.expectEqualStrings("old", parsed.value.object.get("gone").?.array.items[0].object.get("name").?.string);
}

test "a plan registers installed chat artifacts and judges the rest" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", testing.allocator);
    defer testing.allocator.free(root);
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // An index chat model, a legacy one, a decision model, and a file outside the index that does not parse.
    for ([_][]const u8{ "qwen3.8-27b", "gemma-4-12b-qat", "laya" }) |name| {
        const a = catalog.find(name).?;
        const local = try std.fs.path.join(arena, &.{ "models", a.directoryOf(a.main()), a.main().path });
        try tmp.dir.createDirPath(io, std.fs.path.dirname(local).?);
        try tmp.dir.writeFile(io, .{ .sub_path = local, .data = "GGUF" });
        try model.writeSidecar(arena, io, tmp.dir, try model.sidecarPath(arena, local), .{ .repo = a.repo, .file = a.main().path, .revision = a.revision, .sha256 = a.main().sha256, .size = a.main().size, .role = .main, .downloaded_at = "2026-11-12T00:00:00Z", .nuclis_version = "test" });
    }
    try tmp.dir.createDirPath(io, "models/acme/Thing-GGUF");
    try tmp.dir.writeFile(io, .{ .sub_path = "models/acme/Thing-GGUF/Thing-Q4_K_M.gguf", .data = "not a gguf" });
    const registry: config.Models = .{ .entries = &.{.{ .name = "old", .entry = .{ .path = "acme/Gone-GGUF/gone.gguf" } }} };
    const report = try plan(arena, testing.allocator, io, root, registry, "old");
    // Only the current chat artifact needs an entry; it carries the index's pins and verdict.
    try testing.expectEqual(@as(usize, 1), report.registered.len);
    const qwen = report.registered[0];
    try testing.expectEqualStrings("qwen3.8-27b", qwen.name);
    try testing.expectEqualStrings("qwen3.8-27b", qwen.artifact.?);
    try testing.expectEqualStrings("MTP/mtp-Qwen3.8-27B-Q4_0.gguf", qwen.entry.draft.?);
    try testing.expect(qwen.entry.generation.speculative.? and qwen.entry.generation.draft_length.? == 7);
    // The file that does not parse is skipped with its reason; the gone entry is reported.
    try testing.expectEqual(@as(usize, 1), report.skipped.len);
    try testing.expectEqualStrings("acme/Thing-GGUF/Thing-Q4_K_M.gguf", report.skipped[0].path);
    try testing.expectEqual(@as(usize, 1), report.gone.len);
    try testing.expectEqualStrings("old", report.gone[0].name);
    try testing.expectEqualStrings("old", report.model_absent.?);
    try testing.expect(!report.default_installed);
    try testing.expectEqualStrings("qwen3.8-27b", defaultModel(report.registered, report.default_installed));
    // Once registered, a second plan has nothing to add.
    const entries = try toEntries(arena, report.registered);
    const again = try plan(arena, testing.allocator, io, root, .{ .entries = entries }, "qwen3.8-27b");
    try testing.expectEqual(@as(usize, 0), again.registered.len);
    try testing.expect(again.model_absent == null);
}
