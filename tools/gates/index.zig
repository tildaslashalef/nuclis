//! The index as data: `src/catalog.json`, which `make index` writes from
//! `src/catalog.zig` (the runner imports none of nuclis). `resolve` turns
//! each `gates.json` model that names an artifact into the path, projector,
//! and entry the rest of the runner reads; `coverage` names every current
//! chat artifact no gate checks. Everything lives in the caller's arena.
const std = @import("std");
const json = std.json;
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub const path = "src/catalog.json";
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
/// file, a legacy artifact, a literal `path` without the `note` that says
/// why, or neither.
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
        if (std.mem.eql(u8, stringField(a, "status") orelse "", "legacy")) {
            try problems.append(arena, try arena.print("models.{s}: \"{s}\" is legacy: remove what checks it", .{ key, name }));
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

/// Every current chat artifact that no gate's model names: a model nobody
/// checks is not current.
pub fn coverage(arena: Allocator, doc: Value, index: Value) Allocator.Error![]const []const u8 {
    var problems: std.ArrayList([]const u8) = .empty;
    const list = field(index, "artifacts") orelse return problems.items;
    if (list != .array) return problems.items;
    const models = field(doc, "models") orelse return problems.items;
    const gates = field(doc, "gates") orelse return problems.items;
    if (gates != .array) return problems.items;
    for (list.array.items) |a| {
        if (!std.mem.eql(u8, stringField(a, "status") orelse "", "current")) continue;
        if (!std.mem.eql(u8, stringField(a, "kind") orelse "", "generation")) continue;
        const name = stringField(a, "name") orelse continue;
        const checked = for (gates.array.items) |gate| {
            const key = stringField(gate, "model") orelse continue;
            if (std.mem.eql(u8, stringField(field(models, key) orelse .null, "artifact") orelse "", name)) break true;
        } else false;
        if (!checked) try problems.append(arena, try arena.print("{s} is current in {s} but no gate checks it", .{ name, path }));
    }
    return problems.items;
}

/// An index for the tests: one current chat artifact with a projector and
/// two drafts, a legacy one, and a safetensors checkpoint.
pub const sample =
    \\{"schema_version": 1, "artifacts": [
    \\ {"name": "a", "kind": "generation", "status": "current", "defaults": {"draft": "mtp"}, "files": [
    \\  {"role": "main", "variant": null, "local": "o/a/a.gguf"},
    \\  {"role": "mmproj", "variant": null, "local": "o/a/mmproj.gguf"},
    \\  {"role": "draft", "variant": "mtp", "local": "o/a/MTP/mtp.gguf"},
    \\  {"role": "draft", "variant": "dflash2", "local": "o/a/MTP/dflash2.gguf"}]},
    \\ {"name": "old", "kind": "generation", "status": "legacy", "defaults": {"draft": null}, "files": [
    \\  {"role": "main", "variant": null, "local": "o/old/old.gguf"}]},
    \\ {"name": "voice", "kind": "speech", "status": "current", "defaults": {"draft": null}, "files": [
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
        \\ "loose": {"path": "~/x.gguf", "note": "not pinned"}},
        \\ "gates": [{"model": "a"}]}
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

    var bad = try json.parseFromSliceLeaky(Value, a,
        \\{"models": {"old": {"artifact": "old"}, "none": {"artifact": "none"}, "eagle": {"artifact": "a", "file": "draft:eagle"},
        \\ "bare": {"path": "~/x.gguf"}, "empty": {}}, "gates": [{"model": "old"}]}
    , .{});
    const problems = try resolve(a, &bad, index);
    const expected = [_][]const u8{ "\"old\" is legacy", "no artifact \"none\"", "has no file \"draft:eagle\"", "models.bare: a literal path needs a \"note\"", "models.empty: needs an \"artifact\"" };
    for (expected) |needle| {
        for (problems) |problem| {
            if (std.mem.indexOf(u8, problem, needle) != null) break;
        } else {
            std.debug.print("no problem names \"{s}\" in: {f}\n", .{ needle, json.fmt(problems, .{}) });
            return error.TestExpectedProblem;
        }
    }
    const uncovered = try coverage(a, bad, index);
    try testing.expectEqual(@as(usize, 1), uncovered.len);
    try testing.expect(std.mem.indexOf(u8, uncovered[0], "a is current") != null);
}
