//! `nuclis config init` and `nuclis config sync`: registry entries for what
//! is installed under `<root>/models`, and the entries whose file is gone.
//!
//! An installed index artifact of the chat kind gets the entry
//! `config.registryEntry` writes (its pinned files and measured defaults);
//! other kinds resolve by their index name and need none, and `legacy` and
//! `experimental` artifacts are never written. A file outside the index is judged as
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
const kit = @import("ui.zig");
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

/// A key an index entry states that the index now says otherwise: a
/// measured default (verdict, think settings) or a companion file the
/// index now pins and that is on disk. Values are JSON text.
pub const Refresh = struct { name: []const u8, key: []const u8, from: []const u8, to: []const u8 };

/// An index entry whose pinned revision is not the index's: reported, not
/// changed, since the files on disk are the entry's revision.
pub const Behind = struct { name: []const u8, revision: []const u8, index: []const u8 };

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
    /// `config sync`: entry keys set to what the index now says.
    refreshed: []const Refresh = &.{},
    /// `config sync`: keys of the schema the file lacked, added at their
    /// defaults (the registry's entries excepted).
    added: []const []const u8 = &.{},
    behind: []const Behind = &.{},

    /// `config sync`: the file, what changed or would, and a closing line.
    pub fn render(self: Report, arena: Allocator, out: *std.Io.Writer, json: bool, ui: kit.Ui) !void {
        if (json) {
            try std.json.Stringify.value(self, .{ .whitespace = .indent_2, .emit_null_optional_fields = false }, out);
            return out.writeByte('\n');
        }
        try kit.blocks.header(out, ui, try kit.text.clean(arena, self.config_file), if (self.dry_run) "dry run" else "");
        try self.renderRows(arena, out, ui);
        const n = kit.values.count(self.registered.len, "model");
        if (self.registered.len == 0 and (self.refreshed.len > 0 or self.added.len > 0)) {
            try kit.blocks.result(out, ui, if (self.dry_run) .not_checked else .ok, if (self.dry_run) "would update from the catalogue" else "updated from the catalogue", if (self.dry_run) "nothing written" else "nothing new to register");
        } else if (self.registered.len == 0) {
            try kit.blocks.result(out, ui, .not_checked, "nothing to register", try std.fmt.allocPrint(arena, "under {s}", .{try kit.text.clean(arena, self.models_dir)}));
        } else if (self.dry_run) {
            try kit.blocks.result(out, ui, .not_checked, try std.fmt.allocPrint(arena, "would register {f}", .{n}), "nothing written");
        } else try kit.blocks.result(out, ui, .ok, try std.fmt.allocPrint(arena, "registered {f}", .{n}), "");
    }

    /// `config init`: the file and what it registers, the model it runs,
    /// and the next command.
    pub fn renderInit(self: Report, arena: Allocator, out: *std.Io.Writer, ui: kit.Ui, model_name: []const u8) !void {
        try kit.blocks.header(out, ui, try kit.text.clean(arena, self.config_file), if (self.dry_run) "dry run" else "");
        try self.renderRows(arena, out, ui);
        const detail = try std.fmt.allocPrint(arena, "engine.model {s} · {f}", .{ model_name, kit.values.countIrregular(self.registered.len, "entry", "entries") });
        const name = std.fs.path.basename(self.config_file);
        if (self.dry_run) return kit.blocks.result(out, ui, .not_checked, try std.fmt.allocPrint(arena, "would write {s}", .{name}), detail);
        try kit.blocks.result(out, ui, .ok, try std.fmt.allocPrint(arena, "wrote {s}", .{name}), detail);
        if (!self.default_installed and self.registered.len == 0) {
            try kit.blocks.hint(out, ui, "next: ", try std.fmt.allocPrint(arena, "nuclis model pull {s} --all", .{model_name}), " fetches it, then `nuclis config sync` registers it");
        } else try kit.blocks.hint(out, ui, "next: ", "nuclis agent", "");
    }

    /// Each part of the report as a titled table, the notices as `▲`
    /// lines; every part ends with a blank line.
    pub fn renderRows(self: Report, arena: Allocator, out: *std.Io.Writer, ui: kit.Ui) !void {
        const T = kit.table;
        const clean = kit.text.clean;
        if (self.registered.len > 0) {
            try section(out, ui, if (self.dry_run) "Would register" else "Registered");
            const rows = try arena.alloc([]const T.Cell, self.registered.len);
            for (self.registered, rows) |c, *row| {
                var notes: std.ArrayList([]const u8) = .empty;
                if (c.artifact != null) try notes.append(arena, "from the catalogue");
                if (c.adapter) |a| try notes.append(arena, try std.fmt.allocPrint(arena, "{s} adapter", .{a}));
                if (c.entry.profile) |p| try notes.append(arena, try std.fmt.allocPrint(arena, "profile {s}{s}", .{ @tagName(p), if (c.profile_forced) " (forced: the template is not a pinned one)" else "" }));
                if (c.entry.mmproj != null) try notes.append(arena, "mmproj");
                if (c.entry.draft != null) try notes.append(arena, "draft");
                if (c.entry.generation.speculative) |on| try notes.append(arena, if (on) "speculative on" else "speculative off");
                row.* = try arena.dupe(T.Cell, &.{
                    .{ .text = try clean(arena, c.name), .role = .id },
                    .{ .text = catalog.groupName(c.kind) },
                    .{ .text = try clean(arena, c.path) },
                    .{ .text = try clean(arena, try std.mem.join(arena, ", ", notes.items)), .role = .muted },
                });
            }
            try T.render(out, ui, arena, 2, &.{ .{ .title = "NAME", .truncate = false }, .{ .title = "KIND" }, .{ .title = "PATH" }, .{ .title = "NOTE", .drop = 1 } }, rows);
            try out.writeByte('\n');
        }
        if (self.gone.len > 0) {
            try section(out, ui, if (!self.pruned) "Entries whose file is gone" else if (self.dry_run) "Would remove (the file is gone)" else "Removed (the file is gone)");
            const rows = try arena.alloc([]const T.Cell, self.gone.len);
            for (self.gone, rows) |g, *row| row.* = try arena.dupe(T.Cell, &.{ .{ .text = try clean(arena, g.name), .role = .id }, .{ .text = try clean(arena, g.path) } });
            try T.render(out, ui, arena, 2, &.{ .{ .title = "NAME", .truncate = false }, .{ .title = "PATH" } }, rows);
            if (!self.pruned) try kit.blocks.hint(out, ui, "", "nuclis config sync --prune", " removes them");
            try out.writeByte('\n');
        }
        if (self.refreshed.len > 0) {
            try section(out, ui, if (self.dry_run) "Would refresh from the catalogue" else "Refreshed from the catalogue");
            const rows = try arena.alloc([]const T.Cell, self.refreshed.len);
            for (self.refreshed, rows) |r, *row| row.* = try arena.dupe(T.Cell, &.{
                .{ .text = try clean(arena, r.name), .role = .id },
                .{ .text = r.key },
                .{ .text = try clean(arena, r.from), .role = .muted },
                .{ .text = ui.glyphs.arrow, .role = .muted },
                .{ .text = try clean(arena, r.to), .role = .changed },
            });
            try T.render(out, ui, arena, 2, &.{ .{ .title = "NAME", .truncate = false }, .{ .title = "KEY" }, .{ .title = "FROM" }, .{ .title = "" }, .{ .title = "TO" } }, rows);
            try out.writeByte('\n');
        }
        if (self.added.len > 0) {
            try section(out, ui, if (self.dry_run) "Would add at their defaults" else "Added at their defaults");
            for (self.added) |k| try out.print("  {s}{s}{s}\n", .{ ui.on(.id), k, ui.off(.id) });
            try out.writeByte('\n');
        }
        if (self.skipped.len > 0) {
            try section(out, ui, "Skipped");
            const rows = try arena.alloc([]const T.Cell, self.skipped.len);
            for (self.skipped, rows) |sk, *row| row.* = try arena.dupe(T.Cell, &.{ .{ .text = try clean(arena, sk.path) }, .{ .text = try clean(arena, sk.reason), .role = .muted } });
            try T.render(out, ui, arena, 2, &.{ .{ .title = "PATH" }, .{ .title = "REASON", .min = 16 } }, rows);
            try out.writeByte('\n');
        }
        var notices = false;
        for (self.behind) |b| {
            try kit.blocks.result(out, ui, .warning, try clean(arena, b.name), try std.fmt.allocPrint(arena, "pins revision {s}, the catalogue {s}; not changed (the files on disk are the entry's)", .{ b.revision[0..@min(b.revision.len, 12)], b.index[0..@min(b.index.len, 12)] }));
            notices = true;
        }
        if (self.model_absent) |m| {
            try kit.blocks.result(out, ui, .warning, try std.fmt.allocPrint(arena, "engine.model {s} is not installed", .{try clean(arena, m)}), "");
            try kit.blocks.hint(out, ui, "", try std.fmt.allocPrint(arena, "nuclis model pull {s}", .{try clean(arena, m)}), " fetches it, or `nuclis config set engine.model <name>`");
            notices = true;
        }
        if (notices) try out.writeByte('\n');
    }

    fn section(out: *std.Io.Writer, ui: kit.Ui, title: []const u8) !void {
        try out.print("{s}{s}{s}\n", .{ ui.on(.title), title, ui.off(.title) });
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
            if (a.kind != .generation or !a.written() or main.status == .mismatch) continue;
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
    var refreshed: std.ArrayList(Refresh) = .empty;
    var behind: std.ArrayList(Behind) = .empty;
    for (listing.installed) |row| {
        const name = row.artifact orelse continue;
        const as = row.registered orelse continue;
        const e = registry.find(as.name) orelse continue;
        try refreshEntry(arena, as.name, e.*, catalog.find(name).?, row, &refreshed, &behind);
    }
    return .{ .refreshed = refreshed.items, .behind = behind.items, .models_dir = listing.models_dir, .registered = registered.items, .gone = listing.missing, .skipped = skipped.items, .default_installed = listing.find((config.Config.Engine{}).model) != null, .model_absent = if (engine_model) |m| if (listing.find(m) == null) m else null else null };
}

/// The keys `e` states that the index now says otherwise. A null key
/// already follows the index and is left alone; a companion moves only to
/// a file on disk; a draft verdict follows the variant the entry chose.
fn refreshEntry(arena: Allocator, name: []const u8, e: config.ModelEntry, a: *const catalog.Artifact, row: model.Installed, refreshed: *std.ArrayList(Refresh), behind: *std.ArrayList(Behind)) !void {
    if (a.support == .archived) return;
    const fresh = config.registryEntry(a);
    if (e.revision) |r| if (!std.mem.eql(u8, r, a.revision)) try behind.append(arena, .{ .name = name, .revision = r, .index = a.revision });
    if (fresh.mmproj) |p| if (installedAs(row, .mmproj, null)) try note(arena, refreshed, name, "mmproj", e.mmproj, p);
    const default_draft = a.draft(null);
    if (fresh.draft) |p| if (installedAs(row, .draft, default_draft.?.variant)) try note(arena, refreshed, name, "draft", e.draft, p);
    if (a.kind == .generation) {
        const chosen: ?*const catalog.File = if (e.generation.draft) |v| a.draft(v) else default_draft;
        const verdict: catalog.Verdict = if (chosen) |f| f.verdict orelse .{ .speculative = false, .draft_length = 4 } else .{ .speculative = false, .draft_length = 4 };
        try note(arena, refreshed, name, "generation.speculative", e.generation.speculative, verdict.speculative);
        try note(arena, refreshed, name, "generation.draft_length", e.generation.draft_length, verdict.draft_length);
    }
    if (a.defaults.think) |t| try note(arena, refreshed, name, "agent.think", e.agent.think, t);
    if (a.defaults.thinking_budget) |b| try note(arena, refreshed, name, "agent.thinking_budget", e.agent.thinking_budget, b);
}

fn installedAs(row: model.Installed, role: catalog.FileRole, variant: ?[]const u8) bool {
    for (row.files) |f| if (f.role == role) {
        if (variant == null or (f.variant != null and std.mem.eql(u8, f.variant.?, variant.?))) return true;
    };
    return false;
}

/// Records `key` when the entry states it and the index says otherwise.
fn note(arena: Allocator, refreshed: *std.ArrayList(Refresh), name: []const u8, key: []const u8, stated: anytype, index: @typeInfo(@TypeOf(stated)).optional.child) !void {
    const value = stated orelse return;
    const from = try std.json.Stringify.valueAlloc(arena, value, .{});
    const to = try std.json.Stringify.valueAlloc(arena, index, .{});
    if (!std.mem.eql(u8, from, to)) try refreshed.append(arena, .{ .name = name, .key = key, .from = from, .to = to });
}

/// The refreshes as the edits `config.registerEntries` applies.
pub fn toEdits(arena: Allocator, refreshed: []const Refresh) ![]const config.EntryEdit {
    const out = try arena.alloc(config.EntryEdit, refreshed.len);
    for (refreshed, out) |r, *d| d.* = .{ .name = r.name, .key = r.key, .value = r.to };
    return out;
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
    const summary = inference.models.registry.validate(adapter, gpa, doc) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return .{ .rejected = try arena.print("the {s} adapter rejects the file: {s}", .{ @tagName(adapter), @errorName(err) }) },
    };
    if (!summary.inference_available) return .{ .rejected = try arena.print("the {s} adapter binds it, but no executor runs the architecture", .{@tagName(adapter)}) };
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
    try testing.expect(moe.verdict().speculative and moe.verdict().draft_length == 4);
    try testing.expectEqualStrings("gemma-4-12b-qat", familyOf("gemma4", 48).?.name);
    // An unknown layer count, or none, falls back to the architecture's first artifact.
    try testing.expectEqualStrings("gemma-4-12b-qat", familyOf("gemma4", 7).?.name);
    try testing.expectEqualStrings("gemma-4-12b-qat", familyOf("gemma4", null).?.name);
    try testing.expectEqualStrings("qwen3.8-27b", familyOf("qwen35", 64).?.name);
    try testing.expectEqualStrings("qwen3.8-27b", familyOf("qwen35", null).?.name);
    try testing.expect(familyOf("llama", 32) == null);
    // A decision model's GGUF is not a chat artifact.
    try testing.expect(familyOf("clef", null) == null);
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
    try testing.expectEqualStrings("qwen3.8-27b", defaultModel(&.{ speech, chat }, true));
    try testing.expectEqualStrings("gemma-4-e4b-qat", defaultModel(&.{ speech, chat }, false));
    try testing.expectEqualStrings("qwen3.8-27b", defaultModel(&.{speech}, false));
}

test "a report renders both ways from the same rows" {
    const report: Report = .{
        .models_dir = "/m",
        .config_file = "/c.json",
        .registered = &.{.{ .name = "fine", .path = "o/r/f.gguf", .kind = .generation, .architecture = "gemma4", .adapter = "gemma4", .profile_forced = true, .entry = .{ .repo = "o/r", .file = "f.gguf", .revision = "abc", .mmproj = "mmproj.gguf", .profile = .gemma4, .generation = .{ .speculative = false, .draft_length = 4 } } }},
        .gone = &.{.{ .name = "old", .path = "o/gone/x.gguf" }},
        .skipped = &.{.{ .path = "o/r/mmproj.gguf", .reason = "a mmproj companion" }},
    };
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    try report.render(arena, &out.writer, false, .plain);
    try testing.expectEqualStrings(
        \\◆ /c.json
        \\
        \\Registered
        \\  NAME  KIND  PATH        NOTE
        \\  fine  chat  o/r/f.gguf  gemma4 adapter, profile gemma4 (forced: the template is not a pinned one), mmproj, speculative off
        \\
        \\Entries whose file is gone
        \\  NAME  PATH
        \\  old   o/gone/x.gguf
        \\→ nuclis config sync --prune removes them
        \\
        \\Skipped
        \\  PATH             REASON
        \\  o/r/mmproj.gguf  a mmproj companion
        \\
        \\✓ registered 1 model
        \\
    , out.written());
    out.clearRetainingCapacity();
    try report.render(arena, &out.writer, true, .plain);
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
        const local = try std.fs.path.join(arena, &.{ "models", a.repo, a.main().path });
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
    try testing.expectEqualStrings("MTP/Qwen3.8-27B-DFlash2-Q8_0.gguf", qwen.entry.draft.?);
    try testing.expect(qwen.entry.generation.speculative.? and qwen.entry.generation.draft_length.? == 7);
    // The file that does not parse is skipped with its reason; the gone entry is reported.
    try testing.expectEqual(@as(usize, 1), report.skipped.len);
    try testing.expectEqualStrings("acme/Thing-GGUF/Thing-Q4_K_M.gguf", report.skipped[0].path);
    try testing.expectEqual(@as(usize, 1), report.gone.len);
    try testing.expectEqualStrings("old", report.gone[0].name);
    try testing.expectEqualStrings("old", report.model_absent.?);
    try testing.expect(report.default_installed);
    try testing.expectEqualStrings("qwen3.8-27b", defaultModel(report.registered, report.default_installed));
    // Once registered, a second plan has nothing to add.
    const entries = try toEntries(arena, report.registered);
    const again = try plan(arena, testing.allocator, io, root, .{ .entries = entries }, "qwen3.8-27b");
    try testing.expectEqual(@as(usize, 0), again.registered.len);
    try testing.expect(again.model_absent == null);
}

test "sync refreshes what an index entry states and the index now says otherwise" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", testing.allocator);
    defer testing.allocator.free(root);
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const a = catalog.find("qwen3.8-27b").?;
    const local = try std.fs.path.join(arena, &.{ "models", a.repo, a.main().path });
    try tmp.dir.createDirPath(io, std.fs.path.dirname(local).?);
    try tmp.dir.writeFile(io, .{ .sub_path = local, .data = "GGUF" });
    try model.writeSidecar(arena, io, tmp.dir, try model.sidecarPath(arena, local), .{ .repo = a.repo, .file = a.main().path, .revision = a.revision, .sha256 = a.main().sha256, .size = a.main().size, .role = .main, .downloaded_at = "2026-11-12T00:00:00Z", .nuclis_version = "test" });
    // Stale stated values, a key left null, an old pin, and a draft file that is not on disk.
    var entry = config.registryEntry(a);
    entry.revision = "0123456789abcdef";
    entry.draft = "MTP/old-draft.gguf";
    entry.generation.speculative = false;
    entry.generation.draft_length = null;
    const registry: config.Models = .{ .entries = &.{.{ .name = "qwen3.8-27b", .entry = entry }} };
    const report = try plan(arena, testing.allocator, io, root, registry, null);
    try testing.expectEqual(@as(usize, 0), report.registered.len);
    try testing.expectEqual(@as(usize, 1), report.refreshed.len);
    try testing.expectEqualStrings("generation.speculative", report.refreshed[0].key);
    try testing.expectEqualStrings("false", report.refreshed[0].from);
    try testing.expectEqualStrings("true", report.refreshed[0].to);
    try testing.expectEqual(@as(usize, 1), report.behind.len);
    try testing.expectEqualStrings(a.revision, report.behind[0].index);
    const edits = try toEdits(arena, report.refreshed);
    try testing.expectEqualStrings("true", edits[0].value);
}
