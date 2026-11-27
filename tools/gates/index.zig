//! The index as data: `src/catalog.json`, which `make index` writes from
//! `src/catalog.zig` (the runner imports none of nuclis). `resolve` turns
//! each `gates.json` model that names an artifact into the path, projector,
//! and entry the rest of the runner reads; `coverage` names every non-archived
//! artifact no gate checks; `validate` fails closed on an index whose schema,
//! role, or support it does not know. Everything lives in the caller's arena.
const std = @import("std");
const json = std.json;
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub const path = "src/catalog.json";
/// The index schema this runner reads; any other is refused.
pub const schema_version = 2;
const roles = [_][]const u8{ "default", "extra" };
const supports = [_][]const u8{ "verified", "preview", "archived" };
/// Where `nuclis model pull` puts the index's files, as the manifest spells it.
pub const models_root = "~/.nuclis/models/";

fn field(v: Value, name: []const u8) ?Value {
    return switch (v) {
        .object => |o| o.get(name),
        else => null,
    };
}

fn stringField(v: Value, name: []const u8) ?[]const u8 {
    const inner = field(v, name) orelse return null;
    return switch (inner) {
        .string => |s| s,
        else => null,
    };
}

/// The artifact of the index named `name`, or null.
pub fn artifact(index: Value, name: []const u8) ?Value {
    const list = field(index, "artifacts") orelse return null;
    if (list != .array) return null;
    for (list.array.items) |a| {
        if (std.mem.eql(u8, stringField(a, "name") orelse "", name)) return a;
    }
    return null;
}

/// The local path (under the models root) of `selector`'s file: `main`,
/// `mmproj`, `draft` (the default variant), `draft:<variant>`, or
/// `directory` (the main file's directory, for a safetensors checkpoint).
pub fn local(a: Value, selector: []const u8) ?[]const u8 {
    const files = field(a, "files") orelse return null;
    if (files != .array or files.array.items.len == 0) return null;
    if (std.mem.eql(u8, selector, "directory")) return std.fs.path.dirnamePosix(stringField(files.array.items[0], "local") orelse return null);
    const colon = std.mem.indexOfScalar(u8, selector, ':');
    const role = if (colon) |c| selector[0..c] else selector;
    const wanted: ?[]const u8 = if (colon) |c| selector[c + 1 ..] else if (std.mem.eql(u8, role, "draft")) stringField(field(a, "defaults") orelse .null, "draft") else null;
    var first: ?[]const u8 = null;
    for (files.array.items) |f| {
        if (!std.mem.eql(u8, stringField(f, "role") orelse "", role)) continue;
        if (first == null) first = stringField(f, "local");
        const variant = stringField(f, "variant");
        if (wanted) |w| if (variant != null and std.mem.eql(u8, variant.?, w)) return stringField(f, "local");
    }
    return if (colon != null) null else first;
}

/// Rewrites each `models` entry that names an `artifact` into `path`,
/// `mmproj`, and `entry`, and returns the problems: an unknown artifact or
/// file, an archived artifact, a literal `path` without the `note` that
/// says why, or neither.
pub fn resolve(arena: Allocator, doc: *Value, index: Value) Allocator.Error![]const []const u8 {
    var problems: std.ArrayList([]const u8) = .empty;
    const models = (if (doc.* == .object) doc.object.getPtr("models") else null) orelse return problems.items;
    if (models.* != .object) return problems.items;
    for (models.object.keys(), models.object.values()) |key, *entry| {
        if (entry.* != .object) continue;
        const name = stringField(entry.*, "artifact") orelse {
            if (stringField(entry.*, "path") == null) {
                try problems.append(arena, try arena.print("models.{s}: needs an \"artifact\" (or a \"path\" with a \"note\")", .{key}));
            } else if (stringField(entry.*, "note") == null) {
                try problems.append(arena, try arena.print("models.{s}: a literal path needs a \"note\" saying why it is not an index file", .{key}));
            }
            continue;
        };
        if (field(entry.*, "path") != null) try problems.append(arena, try arena.print("models.{s}: names an artifact and a path", .{key}));
        const a = artifact(index, name) orelse {
            try problems.append(arena, try arena.print("models.{s}: no artifact \"{s}\" in {s}", .{ key, name, path }));
            continue;
        };
        if (std.mem.eql(u8, stringField(a, "support") orelse "", "archived")) {
            try problems.append(arena, try arena.print("models.{s}: \"{s}\" is archived: remove what checks it", .{ key, name }));
        }
        const selector = stringField(entry.*, "file") orelse "main";
        const file = local(a, selector) orelse {
            try problems.append(arena, try arena.print("models.{s}: \"{s}\" has no file \"{s}\"", .{ key, name, selector }));
            continue;
        };
        try entry.object.put(arena, "path", .{ .string = try std.mem.concat(arena, u8, &.{ models_root, file }) });
        const whole = std.mem.eql(u8, selector, "main") or std.mem.eql(u8, selector, "directory");
        if (whole) {
            try entry.object.put(arena, "entry", .{ .string = name });
            if (local(a, "mmproj")) |mmproj| try entry.object.put(arena, "mmproj", .{ .string = try std.mem.concat(arena, u8, &.{ models_root, mmproj }) });
        }
    }
    return problems.items;
}

fn oneOf(value: ?[]const u8, allowed: []const []const u8) bool {
    const v = value orelse return false;
    for (allowed) |a| if (std.mem.eql(u8, v, a)) return true;
    return false;
}

/// The index's own problems: a `schema_version` other than this runner's,
/// no `artifacts` list, or an artifact whose `role` or `support` is missing
/// or unknown. Unknown is refused, never read as verified or ungated.
pub fn validate(arena: Allocator, index: Value) Allocator.Error![]const []const u8 {
    var problems: std.ArrayList([]const u8) = .empty;
    const version = field(index, "schema_version") orelse Value.null;
    if (version != .integer or version.integer != schema_version)
        try problems.append(arena, try arena.print("{s}: schema_version must be {d}; run make index", .{ path, schema_version }));
    const list = field(index, "artifacts") orelse Value.null;
    if (list != .array) {
        try problems.append(arena, try arena.print("{s}: no \"artifacts\" list", .{path}));
        return problems.items;
    }
    for (list.array.items, 0..) |a, i| {
        const name = stringField(a, "name") orelse try arena.print("artifacts[{d}]", .{i});
        if (!oneOf(stringField(a, "role"), &roles))
            try problems.append(arena, try arena.print("{s}: {s} has no known role (default or extra)", .{ path, name }));
        if (!oneOf(stringField(a, "support"), &supports))
            try problems.append(arena, try arena.print("{s}: {s} has no known support (verified, preview, or archived)", .{ path, name }));
    }
    return problems.items;
}

/// Every artifact of any kind that is not archived (verified or preview,
/// default or extra) and that no gate's model names: a model nobody checks
/// is neither. One of unknown support needs a gate too, so `validate`'s
/// refusal is not the only line.
pub fn coverage(arena: Allocator, doc: Value, index: Value) Allocator.Error![]const []const u8 {
    var problems: std.ArrayList([]const u8) = .empty;
    const list = field(index, "artifacts") orelse return problems.items;
    if (list != .array) return problems.items;
    const models = field(doc, "models") orelse return problems.items;
    const gates = field(doc, "gates") orelse return problems.items;
    if (gates != .array) return problems.items;
    for (list.array.items) |a| {
        const support = stringField(a, "support") orelse "of no known support";
        if (std.mem.eql(u8, support, "archived")) continue;
        const name = stringField(a, "name") orelse "an unnamed artifact";
        const checked = for (gates.array.items) |gate| {
            const key = stringField(gate, "model") orelse continue;
            if (std.mem.eql(u8, stringField(field(models, key) orelse .null, "artifact") orelse "", name)) break true;
        } else false;
        if (!checked) try problems.append(arena, try arena.print("{s} is {s} in {s} but no gate checks it", .{ name, support, path }));
    }
    return problems.items;
}

/// An index for the tests: one verified default chat artifact with a
/// projector and two drafts, a preview extra, an archived extra, and a
/// safetensors checkpoint.
pub const sample =
    \\{"schema_version": 2, "artifacts": [
    \\ {"name": "a", "kind": "generation", "role": "default", "support": "verified", "defaults": {"draft": "mtp"}, "files": [
    \\  {"role": "main", "variant": null, "local": "o/a/a.gguf"},
    \\  {"role": "mmproj", "variant": null, "local": "o/a/mmproj.gguf"},
    \\  {"role": "draft", "variant": "mtp", "local": "o/a/MTP/mtp.gguf"},
    \\  {"role": "draft", "variant": "dflash2", "local": "o/a/MTP/dflash2.gguf"}]},
    \\ {"name": "trial", "kind": "generation", "role": "extra", "support": "preview", "defaults": {"draft": null}, "files": [
    \\  {"role": "main", "variant": null, "local": "o/trial/trial.gguf"}]},
    \\ {"name": "old", "kind": "generation", "role": "extra", "support": "archived", "defaults": {"draft": null}, "files": [
    \\  {"role": "main", "variant": null, "local": "o/old/old.gguf"}]},
    \\ {"name": "voice", "kind": "speech", "role": "default", "support": "verified", "defaults": {"draft": null}, "files": [
    \\  {"role": "main", "variant": null, "local": "o/voice/model.safetensors"}]}
    \\]}
;

const testing = std.testing;

test "artifacts resolve to paths, and every mistake is named" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const index = try json.parseFromSliceLeaky(Value, a, sample, .{});
    var doc = try json.parseFromSliceLeaky(Value, a,
        \\{"models": {
        \\ "a": {"artifact": "a"}, "a_mtp": {"artifact": "a", "file": "draft"},
        \\ "a_dflash2": {"artifact": "a", "file": "draft:dflash2"}, "voice": {"artifact": "voice", "file": "directory"},
        \\ "trial": {"artifact": "trial"}, "loose": {"path": "~/x.gguf", "note": "not pinned"}},
        \\ "gates": [{"model": "a"}, {"model": "trial"}, {"model": "voice"}]}
    , .{});
    try testing.expectEqual(@as(usize, 0), (try resolve(a, &doc, index)).len);
    const models = doc.object.get("models").?;
    try testing.expectEqualStrings("~/.nuclis/models/o/a/a.gguf", stringField(models.object.get("a").?, "path").?);
    try testing.expectEqualStrings("~/.nuclis/models/o/a/mmproj.gguf", stringField(models.object.get("a").?, "mmproj").?);
    try testing.expectEqualStrings("a", stringField(models.object.get("a").?, "entry").?);
    try testing.expectEqualStrings("~/.nuclis/models/o/a/MTP/mtp.gguf", stringField(models.object.get("a_mtp").?, "path").?);
    try testing.expect(stringField(models.object.get("a_mtp").?, "entry") == null);
    try testing.expectEqualStrings("~/.nuclis/models/o/a/MTP/dflash2.gguf", stringField(models.object.get("a_dflash2").?, "path").?);
    try testing.expectEqualStrings("~/.nuclis/models/o/voice", stringField(models.object.get("voice").?, "path").?);
    try testing.expectEqual(@as(usize, 0), (try coverage(a, doc, index)).len);
    try testing.expectEqual(@as(usize, 0), (try validate(a, index)).len);

    var bad = try json.parseFromSliceLeaky(Value, a,
        \\{"models": {"old": {"artifact": "old"}, "none": {"artifact": "none"}, "eagle": {"artifact": "a", "file": "draft:eagle"},
        \\ "bare": {"path": "~/x.gguf"}, "empty": {}}, "gates": [{"model": "old"}]}
    , .{});
    const problems = try resolve(a, &bad, index);
    const expected = [_][]const u8{ "\"old\" is archived", "no artifact \"none\"", "has no file \"draft:eagle\"", "models.bare: a literal path needs a \"note\"", "models.empty: needs an \"artifact\"" };
    for (expected) |needle| {
        for (problems) |problem| {
            if (std.mem.indexOf(u8, problem, needle) != null) break;
        } else {
            std.debug.print("no problem names \"{s}\" in: {f}\n", .{ needle, json.fmt(problems, .{}) });
            return error.TestExpectedProblem;
        }
    }
    // Every kind is covered: a preview artifact is gated like a verified one,
    // a speech one like a chat one; an archived one never needs a gate.
    const uncovered = try coverage(a, bad, index);
    try testing.expectEqual(@as(usize, 3), uncovered.len);
    try testing.expect(std.mem.indexOf(u8, uncovered[0], "a is verified") != null);
    try testing.expect(std.mem.indexOf(u8, uncovered[1], "trial is preview") != null);
    try testing.expect(std.mem.indexOf(u8, uncovered[2], "voice is verified") != null);
}

test "an index of another schema, or with an unknown role or support, is refused" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { index: []const u8, needle: []const u8 }{
        .{ .index = "{\"artifacts\": []}", .needle = "schema_version must be 2" },
        .{ .index = "{\"schema_version\": 1, \"artifacts\": []}", .needle = "schema_version must be 2" },
        .{ .index = "{\"schema_version\": \"2\", \"artifacts\": []}", .needle = "schema_version must be 2" },
        .{ .index = "{\"schema_version\": 2}", .needle = "no \"artifacts\" list" },
        .{ .index = "{\"schema_version\": 2, \"artifacts\": [{\"name\": \"x\", \"support\": \"verified\"}]}", .needle = "x has no known role" },
        .{ .index = "{\"schema_version\": 2, \"artifacts\": [{\"name\": \"x\", \"role\": \"current\", \"support\": \"verified\"}]}", .needle = "x has no known role" },
        .{ .index = "{\"schema_version\": 2, \"artifacts\": [{\"name\": \"x\", \"role\": \"extra\"}]}", .needle = "x has no known support" },
        .{ .index = "{\"schema_version\": 2, \"artifacts\": [{\"name\": \"x\", \"role\": \"extra\", \"support\": \"legacy\"}]}", .needle = "x has no known support" },
    };
    for (cases) |case| {
        const index = try json.parseFromSliceLeaky(Value, a, case.index, .{});
        const problems = try validate(a, index);
        try testing.expectEqual(@as(usize, 1), problems.len);
        try testing.expect(std.mem.indexOf(u8, problems[0], case.needle) != null);
    }
    // An artifact of unknown support is not taken as archived: it needs a gate too.
    const unknown = try json.parseFromSliceLeaky(Value, a, "{\"schema_version\": 2, \"artifacts\": [{\"name\": \"x\", \"role\": \"extra\", \"support\": \"legacy\"}]}", .{});
    const doc = try json.parseFromSliceLeaky(Value, a, "{\"models\": {}, \"gates\": []}", .{});
    const uncovered = try coverage(a, doc, unknown);
    try testing.expectEqual(@as(usize, 1), uncovered.len);
    try testing.expect(std.mem.indexOf(u8, uncovered[0], "x is legacy") != null);
}
