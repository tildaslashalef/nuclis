//! `nuclis config init --discover`: registry entries for the runnable GGUF
//! files under `<root>/models` that neither the catalogue nor the registry
//! names (a finetune pulled by repository, a file copied in by hand).
//!
//! Each main file is judged as `model inspect` judges a directory (an
//! adapter for its architecture, every tensor in the executable set, the
//! adapter's binding), its profile is chosen by template digest and forced
//! to the family's when no profile matches (the finetune case), and the
//! companions beside it fill `mmproj` / `mtp`. Decisions are pure over the
//! listing; `config.registerDiscovered` writes the file.
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
    architecture: []const u8,
    adapter: []const u8,
    profile: config.Profile,
    /// The template digest matched no profile, so the family's is forced.
    profile_forced: bool,
    /// The sidecar's facts when the file was pulled by nuclis; the entry is
    /// then `repo` + `file` + `revision`, else `path`.
    repo: ?[]const u8,
    file: ?[]const u8,
    revision: ?[]const u8,
    mmproj: ?[]const u8,
    mtp: ?[]const u8,
    speculative: ?bool,
    draft_length: ?usize,
};

pub const Skipped = struct { path: []const u8, reason: []const u8 };

pub const Report = struct {
    schema_version: u32 = 1,
    models_dir: []const u8,
    config_file: []const u8 = "",
    dry_run: bool = false,
    written: bool = false,
    registered: []const Candidate,
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
            try out.print("{s}dry run:{s} {d} file(s) under {s}{s}{s} would be registered in {s}{s}{s}; nothing written", .{ sty.on(.bold), off, self.registered.len, sty.on(.code), self.models_dir, off, sty.on(.code), self.config_file, off });
        } else {
            try out.print("{s}registered{s} {d} file(s) under {s}{s}{s} in {s}{s}{s}", .{ sty.on(.success), off, self.registered.len, sty.on(.code), self.models_dir, off, sty.on(.code), self.config_file, off });
        }
        try out.writeByte('\n');
        for (self.registered) |c| {
            try out.print("  {s}{s}{s}  {s}{s}{s}\n", .{ sty.on(.keyword), c.name, off, sty.on(.code), c.path, off });
            try out.print("      {s}{s} adapter, profile {s}{s}", .{ sty.on(.dim), c.adapter, @tagName(c.profile), if (c.profile_forced) " (forced: the template is not a pinned one)" else "" });
            if (c.mmproj) |m| try out.print(", mmproj {s}", .{m});
            if (c.mtp) |m| try out.print(", mtp {s}", .{m});
            if (c.speculative) |s| try out.print(", speculative {s}", .{if (s) "on" else "off"});
            try out.print("{s}\n", .{off});
        }
        if (self.skipped.len > 0) {
            try out.print("{s}skipped{s}\n", .{ sty.on(.header), off });
            for (self.skipped) |s| try out.print("  {s}{s}{s}  {s}{s}{s}\n", .{ sty.on(.code), s.path, off, sty.on(.dim), s.reason, off });
        }
        if (self.registered.len > 0 and !self.dry_run) {
            try out.print("{s}next:{s} {s}nuclis agent --model {s}{s}, or {s}nuclis config set engine.model {s}{s} to make it the default\n", .{ sty.on(.bold), off, sty.on(.code), self.registered[0].name, off, sty.on(.code), self.registered[0].name, off });
        }
    }
};

/// The listing's unnamed files judged one by one; every decision lands in
/// the report as a candidate or a skipped row with its reason.
pub fn discover(arena: Allocator, gpa: Allocator, io: std.Io, root: []const u8, registry: config.Models) !Report {
    const listing = try model.list(arena, io, root, registry);
    var registered: std.ArrayList(Candidate) = .empty;
    var skipped: std.ArrayList(Skipped) = .empty;
    var taken: std.ArrayList([]const u8) = .empty;
    for (registry.entries) |e| try taken.append(arena, e.name);
    for (&catalog.entries) |*e| try taken.append(arena, e.name);
    for (listing.other) |f| {
        if (f.registered) |r| {
            try skipped.append(arena, .{ .path = f.path, .reason = try std.fmt.allocPrint(arena, "already registered as {s}", .{r.name}) });
            continue;
        }
        if (companionRole(f)) |role| {
            try skipped.append(arena, .{ .path = f.path, .reason = try std.fmt.allocPrint(arena, "a {s} companion; it fills the entry of the main file beside it", .{@tagName(role)}) });
            continue;
        }
        const absolute = try std.fs.path.join(arena, &.{ listing.models_dir, f.path });
        var doc = inference.gguf.open(gpa, io, absolute, .{}) catch |err| {
            try skipped.append(arena, .{ .path = f.path, .reason = try std.fmt.allocPrint(arena, "the GGUF directory does not parse: {s}", .{@errorName(err)}) });
            continue;
        };
        defer doc.deinit();
        const architecture = try arena.dupe(u8, doc.string("general.architecture") orelse "");
        const adapter = switch (try judge(arena, gpa, &doc, architecture)) {
            .adapter => |a| a,
            .rejected => |reason| {
                try skipped.append(arena, .{ .path = f.path, .reason = reason });
                continue;
            },
        };
        const family = familyOf(architecture) orelse {
            try skipped.append(arena, .{ .path = f.path, .reason = try std.fmt.allocPrint(arena, "no catalogue family for architecture \"{s}\"", .{architecture}) });
            continue;
        };
        const pinned = inference.profiles.forDocument(doc);
        const mmproj = companionBeside(listing.other, f.path, .mmproj) orelse catalogCompanionBeside(listing.catalog, f.path, .mmproj);
        const mtp = companionBeside(listing.other, f.path, .mtp) orelse catalogCompanionBeside(listing.catalog, f.path, .mtp);
        const name = try uniqueName(arena, f.path, taken.items);
        try taken.append(arena, name);
        // The sidecar names the pull; the entry then locates the file the way
        // `model pull --register` would, so `model pull <name>` works on it.
        const pulled = f.sidecar != null and f.sidecar.?.role == .main and std.mem.eql(u8, f.path, try std.fs.path.join(arena, &.{ f.sidecar.?.repo, f.sidecar.?.file }));
        // The family's measured verdict, except that a family whose drafter
        // is a companion file has nothing to draft with when it is absent.
        const needs_companion = family.companion(.mtp) != null;
        try registered.append(arena, .{
            .name = name,
            .path = f.path,
            .architecture = architecture,
            .adapter = @tagName(adapter),
            .profile = pinned orelse family.profile,
            .profile_forced = pinned == null,
            .repo = if (pulled) f.sidecar.?.repo else null,
            .file = if (pulled) f.sidecar.?.file else null,
            .revision = if (pulled) f.sidecar.?.revision else null,
            .mmproj = mmproj,
            .mtp = mtp,
            .speculative = family.speculative and (mtp != null or !needs_companion),
            .draft_length = family.draft_length,
        });
    }
    return .{ .models_dir = listing.models_dir, .registered = registered.items, .skipped = skipped.items };
}

const Judgement = union(enum) { adapter: inference.models.Adapter, rejected: []const u8 };

/// `model inspect`'s three checks on a local directory: an adapter for the
/// architecture, every tensor in its executable set, the binding.
fn judge(arena: Allocator, gpa: Allocator, doc: *const inference.gguf.Document, architecture: []const u8) !Judgement {
    const adapter = inference.models.adapterFor(architecture) orelse return .{ .rejected = if (architecture.len == 0)
        "the file declares no general.architecture"
    else
        try std.fmt.allocPrint(arena, "no adapter for architecture \"{s}\"", .{architecture}) };
    for (doc.tensors) |tensor| if (!inference.models.registry.executableEncoding(adapter, tensor.encoding_id)) {
        const layout = inference.encoding.layout(tensor.encoding_id);
        return .{ .rejected = try std.fmt.allocPrint(arena, "tensor {s} uses encoding {s} (id {d}), outside the {s} adapter's executable set", .{ tensor.name, if (layout) |l| l.name else "?", tensor.encoding_id, @tagName(adapter) }) };
    };
    _ = inference.models.registry.validate(adapter, gpa, doc) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return .{ .rejected = try std.fmt.allocPrint(arena, "the {s} adapter rejects the file: {s}", .{ @tagName(adapter), @errorName(err) }) },
    };
    return .{ .adapter = adapter };
}

/// The catalogue entry that carries the family's profile and verdict.
fn familyOf(architecture: []const u8) ?*const catalog.Entry {
    for (&catalog.entries) |*e| if (std.mem.eql(u8, e.architecture, architecture)) return e;
    return null;
}

/// A companion by its sidecar's role, else by the names the Hub uses
/// (`mmproj-*.gguf`, `mtp-*.gguf`, `dflash*.gguf`, `*imatrix*`).
pub fn companionRole(f: model.Listed) ?model.Role {
    if (f.sidecar) |s| if (s.role != .main) return s.role;
    return companionRoleOfName(std.fs.path.basename(f.path));
}

fn companionRoleOfName(name: []const u8) ?model.Role {
    var lower: [256]u8 = undefined;
    const n = @min(name.len, lower.len);
    const l = std.ascii.lowerString(lower[0..n], name[0..n]);
    if (std.mem.indexOf(u8, l, "mmproj") != null) return .mmproj;
    if (std.mem.startsWith(u8, l, "mtp") or std.mem.indexOf(u8, l, "-mtp") != null or std.mem.indexOf(u8, l, "dflash") != null) return .mtp;
    if (std.mem.indexOf(u8, l, "imatrix") != null) return .imatrix;
    return null;
}

/// The first companion of `role` in the same directory as `main`, as the
/// file name an entry records (relative to that directory).
fn companionBeside(files: []const model.Listed, main: []const u8, role: model.Role) ?[]const u8 {
    const dir = std.fs.path.dirname(main) orelse "";
    for (files) |f| {
        if (std.mem.eql(u8, f.path, main)) continue;
        // The same directory or one below it (`MTP/mtp-….gguf` in the Qwen layout).
        const d = std.fs.path.dirname(f.path) orelse "";
        if (!std.mem.eql(u8, d, dir) and !(dir.len == 0 or (std.mem.startsWith(u8, d, dir) and d.len > dir.len and d[dir.len] == '/'))) continue;
        if (companionRole(f) == role) return f.path[dir.len + @as(usize, if (dir.len > 0) 1 else 0) ..];
    }
    return null;
}

/// A catalogue entry's companion in the same directory (a second main file
/// dropped beside a pulled entry shares its projector).
fn catalogCompanionBeside(rows: []const model.CatalogRow, main: []const u8, role: model.Role) ?[]const u8 {
    const dir = std.fs.path.dirname(main) orelse "";
    for (rows) |row| for (row.companions) |c| {
        if (c.role != role or c.status == .absent) continue;
        const d = std.fs.path.dirname(c.path) orelse "";
        if (std.mem.eql(u8, d, dir)) return c.path[dir.len + @as(usize, if (dir.len > 0) 1 else 0) ..];
    };
    return null;
}

/// The repository's name as an entry name: lower-case, `-gguf` dropped,
/// `[a-z0-9._-]` kept (other bytes become `-`); a taken name gains the
/// file's quantization suffix, then a counter. Never a catalogue name.
pub fn uniqueName(arena: Allocator, path: []const u8, taken: []const []const u8) ![]const u8 {
    var parts = std.mem.splitScalar(u8, path, '/');
    _ = parts.next(); // owner
    const repo = parts.next() orelse path;
    const base = try sanitize(arena, repo);
    if (!isTaken(base, taken)) return base;
    const stem = std.fs.path.stem(std.fs.path.basename(path));
    const quant = if (std.mem.lastIndexOfScalar(u8, stem, '-')) |i| stem[i + 1 ..] else stem;
    const with_quant = try std.fmt.allocPrint(arena, "{s}-{s}", .{ base, try sanitize(arena, quant) });
    if (!isTaken(with_quant, taken)) return with_quant;
    var n: usize = 2;
    while (n < 100) : (n += 1) {
        const numbered = try std.fmt.allocPrint(arena, "{s}-{d}", .{ with_quant, n });
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

/// The candidates in the form `config.registerDiscovered` writes.
pub fn toEntries(arena: Allocator, candidates: []const Candidate) ![]const config.Discovered {
    const out = try arena.alloc(config.Discovered, candidates.len);
    for (candidates, out) |c, *d| d.* = .{
        .name = c.name,
        .path = if (c.repo == null) c.path else null,
        .repo = c.repo,
        .file = c.file,
        .revision = c.revision,
        .mmproj = c.mmproj,
        .mtp = c.mtp,
        .profile = if (c.profile_forced) c.profile else null,
        .speculative = c.speculative,
        .draft_length = c.draft_length,
    };
    return out;
}

// ----- tests -----

const testing = std.testing;

test "companions are told by sidecar role, then by the Hub's names" {
    const none: ?model.Sidecar = null;
    try testing.expectEqual(model.Role.mmproj, companionRole(.{ .path = "o/r/mmproj-BF16.gguf", .size = 1, .sidecar = none }).?);
    try testing.expectEqual(model.Role.mtp, companionRole(.{ .path = "o/r/MTP/mtp-x.gguf", .size = 1, .sidecar = none }).?);
    try testing.expectEqual(model.Role.mtp, companionRole(.{ .path = "o/r/dflash-kquant.gguf", .size = 1, .sidecar = none }).?);
    try testing.expect(companionRole(.{ .path = "o/r/Model-Q4_K_M.gguf", .size = 1, .sidecar = none }) == null);
    const sidecar: model.Sidecar = .{ .repo = "o/r", .file = "odd-name.gguf", .revision = "x", .sha256 = "y", .size = 1, .role = .imatrix, .downloaded_at = "", .nuclis_version = "" };
    try testing.expectEqual(model.Role.imatrix, companionRole(.{ .path = "o/r/odd-name.gguf", .size = 1, .sidecar = sidecar }).?);
}

test "a companion is found beside its main file and named relative to the directory" {
    const none: ?model.Sidecar = null;
    const files = [_]model.Listed{
        .{ .path = "o/r/Main-Q4_K_M.gguf", .size = 1, .sidecar = none },
        .{ .path = "o/r/mmproj-BF16.gguf", .size = 1, .sidecar = none },
        .{ .path = "o/r/MTP/mtp-main.gguf", .size = 1, .sidecar = none },
        .{ .path = "o/other/mmproj-BF16.gguf", .size = 1, .sidecar = none },
    };
    try testing.expectEqualStrings("mmproj-BF16.gguf", companionBeside(&files, "o/r/Main-Q4_K_M.gguf", .mmproj).?);
    try testing.expectEqualStrings("MTP/mtp-main.gguf", companionBeside(&files, "o/r/Main-Q4_K_M.gguf", .mtp).?);
    try testing.expect(companionBeside(&files, "o/other/x.gguf", .mtp) == null);
}

test "names come from the repository, never collide, and never take a catalogue name" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try testing.expectEqualStrings("gemma4-12b-qat-uncensored-hauhaucs-balanced", try uniqueName(arena, "HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf", &.{}));
    try testing.expectEqualStrings("ternary-bonsai-2-27b", try uniqueName(arena, "prism-ml/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf", &.{}));
    try testing.expectEqualStrings("ternary-bonsai-2-27b-pq2_0", try uniqueName(arena, "prism-ml/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf", &.{"ternary-bonsai-2-27b"}));
    try testing.expectEqualStrings("ternary-bonsai-2-27b-pq2_0-2", try uniqueName(arena, "prism-ml/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf", &.{ "ternary-bonsai-2-27b", "ternary-bonsai-2-27b-pq2_0" }));
    try testing.expectEqualStrings("a-b_c", try uniqueName(arena, "o/A  b_c.GGUF/x.gguf", &.{}));
}

test "a report renders both ways from the same rows" {
    const report: Report = .{
        .models_dir = "/m",
        .config_file = "/c.json",
        .registered = &.{.{ .name = "fine", .path = "o/r/f.gguf", .architecture = "gemma4", .adapter = "gemma4", .profile = .gemma4, .profile_forced = true, .repo = "o/r", .file = "f.gguf", .revision = "abc", .mmproj = "mmproj.gguf", .mtp = null, .speculative = false, .draft_length = 4 }},
        .skipped = &.{.{ .path = "o/r/mmproj.gguf", .reason = "a mmproj companion" }},
    };
    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    try report.render(&out.writer, false, .none);
    try testing.expect(std.mem.indexOf(u8, out.written(), "registered 1 file(s)") != null);
    try testing.expect(std.mem.indexOf(u8, out.written(), "profile gemma4 (forced") != null);
    try testing.expect(std.mem.indexOf(u8, out.written(), "nuclis agent --model fine") != null);
    out.clearRetainingCapacity();
    try report.render(&out.writer, true, .none);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, out.written(), .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("fine", parsed.value.object.get("registered").?.array.items[0].object.get("name").?.string);
    try testing.expect(parsed.value.object.get("registered").?.array.items[0].object.get("mtp") == null);
}
