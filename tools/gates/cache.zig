//! Remembered gates. A pass is stored under `$NUCLIS_HOME/gates/passes.json`
//! (default `~/.nuclis`) by a key over what the gate's result depends on: its
//! definition, the fingerprints of every engine file not exclusive to other
//! families, its model files' identity, and the optimize mode. Only passes
//! are stored; a failure or a skip always runs again. `times.json` beside it
//! holds each gate's last seconds, which `select.byCost` orders by.
const std = @import("std");
const json = std.json;
const manifest = @import("manifest.zig");
const select = @import("select.zig");
const tree_module = @import("tree.zig");
const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Gate = manifest.Gate;

pub const Key = [Sha256.digest_length * 2]u8;

/// Passes kept; the oldest go first. Branches alternate, so a key is not
/// replaced when its gate passes under another.
const max_passes = 4096;
const max_store_bytes = 64 << 20;
/// A model directory past this is not a model.
const max_model_files = 1 << 16;

/// The canonical form of a JSON value: sorted keys, minified.
fn writeCanonical(arena: Allocator, w: *std.Io.Writer, value: json.Value, skip: []const []const u8) (Allocator.Error || std.Io.Writer.Error)!void {
    switch (value) {
        .object => |object| {
            const keys = try arena.dupe([]const u8, object.keys());
            std.mem.sort([]const u8, keys, {}, struct {
                fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                    return std.mem.lessThan(u8, a, b);
                }
            }.lessThan);
            try w.writeByte('{');
            var first = true;
            next: for (keys) |name| {
                for (skip) |skipped| {
                    if (std.mem.eql(u8, skipped, name)) continue :next;
                }
                if (!first) try w.writeByte(',');
                first = false;
                try json.Stringify.value(name, .{}, w);
                try w.writeByte(':');
                try writeCanonical(arena, w, object.get(name).?, &.{});
            }
            try w.writeByte('}');
        },
        .array => |array| {
            try w.writeByte('[');
            for (array.items, 0..) |item, i| {
                if (i > 0) try w.writeByte(',');
                try writeCanonical(arena, w, item, &.{});
            }
            try w.writeByte(']');
        },
        else => try json.Stringify.value(value, .{}, w),
    }
}

/// The gate's definition as a key input: what it runs and checks, not the
/// prose around it.
pub fn canonicalGate(arena: Allocator, gate: *const Gate) Allocator.Error![]const u8 {
    var out: std.Io.Writer.Allocating = .init(arena);
    writeCanonical(arena, &out.writer, gate.raw, &.{ "note", "evidence" }) catch return error.OutOfMemory;
    return out.written();
}

/// The optimize mode of the build a gate runs.
pub fn optimizeOf(m: *const manifest.Manifest, gate: *const Gate) []const u8 {
    for (gate.command) |arg| {
        if (std.mem.indexOf(u8, arg, "-cpu}") != null) return m.cpu_optimize;
    }
    return m.metal_optimize;
}

pub fn key(
    arena: Allocator,
    m: *const manifest.Manifest,
    tree: *const tree_module.Tree,
    gate: *const Gate,
    model_identity: []const u8,
) Allocator.Error!Key {
    const bit = @as(u64, 1) << (m.familyIndex(gate.family) orelse 0);
    var hasher = Sha256.init(.{});
    hasher.update(try canonicalGate(arena, gate));
    hasher.update(&.{0});
    hasher.update(optimizeOf(m, gate));
    hasher.update(&.{0});
    for (tree.files) |file| {
        if (file.families != 0 and file.families & bit == 0) continue;
        hasher.update(file.path);
        hasher.update(&.{0});
        hasher.update(&file.digest);
    }
    hasher.update(model_identity);
    return std.fmt.bytesToHex(hasher.finalResult(), .lower);
}

fn appendIdentity(arena: Allocator, lines: *std.ArrayList([]const u8), path: []const u8, stat: std.Io.File.Stat) Allocator.Error!void {
    try lines.append(arena, try arena.print("{s} {d} {d}\n", .{ path, stat.size, stat.mtime.nanoseconds }));
}

/// Path, size, and mtime of each file in `paths`, a directory by its files.
pub fn modelIdentity(arena: Allocator, io: std.Io, paths: []const []const u8) ![]const u8 {
    var lines: std.ArrayList([]const u8) = .empty;
    const cwd = std.Io.Dir.cwd();
    for (paths) |path| {
        const stat = try cwd.statFile(io, path, .{});
        if (stat.kind != .directory) {
            try appendIdentity(arena, &lines, path, stat);
            continue;
        }
        var dir = try cwd.openDir(io, path, .{ .iterate = true });
        defer dir.close(io);
        var walker = try dir.walk(arena);
        defer walker.deinit();
        var count: usize = 0;
        while (try walker.next(io)) |entry| {
            if (entry.kind == .directory) continue;
            count += 1;
            if (count > max_model_files) return error.ModelTooLarge;
            // A broken link is not a model file.
            const inner = entry.dir.statFile(io, entry.basename, .{}) catch continue;
            try appendIdentity(arena, &lines, try std.fs.path.join(arena, &.{ path, entry.path }), inner);
        }
    }
    std.mem.sort([]const u8, lines.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lessThan);
    return std.mem.concat(arena, u8, lines.items);
}

pub const Pass = struct {
    gate: []const u8,
    revision: []const u8,
    seconds: f64,
};

/// `$NUCLIS_HOME/gates`, or `~/.nuclis/gates`.
pub fn directory(arena: Allocator, env: *const std.process.Environ.Map) error{ OutOfMemory, InvalidHome, NoHome }![]const u8 {
    if (env.get("NUCLIS_HOME")) |home| {
        if (home.len == 0 or !std.fs.path.isAbsolute(home)) return error.InvalidHome;
        return std.fs.path.join(arena, &.{ home, "gates" });
    }
    const home = env.get("HOME") orelse return error.NoHome;
    return std.fs.path.join(arena, &.{ home, ".nuclis", "gates" });
}

fn readJson(arena: Allocator, io: std.Io, path: []const u8) ?json.Value {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(max_store_bytes)) catch return null;
    return json.parseFromSliceLeaky(json.Value, arena, bytes, .{}) catch null;
}

/// Replaces `path` whole, so a reader never sees half a file.
fn writeAtomic(io: std.Io, path: []const u8, bytes: []const u8) !void {
    const cwd = std.Io.Dir.cwd();
    var file = try cwd.createFileAtomic(io, path, .{ .make_path = true, .replace = true });
    defer file.deinit(io);
    try file.file.writeStreamingAll(io, bytes);
    try file.replace(io);
}

pub const Passes = struct {
    path: []const u8,
    /// Insertion order is age.
    entries: std.array_hash_map.String(Pass) = .empty,

    pub fn load(arena: Allocator, io: std.Io, dir: []const u8) Allocator.Error!Passes {
        var passes: Passes = .{ .path = try std.fs.path.join(arena, &.{ dir, "passes.json" }) };
        const doc = readJson(arena, io, passes.path) orelse return passes;
        if (doc != .object) return passes;
        const stored = doc.object.get("passes") orelse return passes;
        if (stored != .object) return passes;
        for (stored.object.keys(), stored.object.values()) |hex, entry| {
            if (entry != .object) continue;
            const gate = entry.object.get("gate") orelse continue;
            const revision = entry.object.get("revision") orelse continue;
            const seconds = entry.object.get("seconds") orelse continue;
            if (gate != .string or revision != .string) continue;
            try passes.entries.put(arena, hex, .{
                .gate = gate.string,
                .revision = revision.string,
                .seconds = switch (seconds) {
                    .float => |f| f,
                    .integer => |i| @floatFromInt(i),
                    else => continue,
                },
            });
        }
        return passes;
    }

    pub fn get(passes: *const Passes, hex: []const u8) ?Pass {
        return passes.entries.get(hex);
    }

    pub fn put(passes: *Passes, arena: Allocator, hex: []const u8, pass: Pass) Allocator.Error!void {
        try passes.entries.put(arena, try arena.dupe(u8, hex), pass);
    }

    /// Merges into what is on disk, because another worktree may share the
    /// store, keeps the newest `max_passes`, and writes.
    pub fn save(passes: *const Passes, arena: Allocator, io: std.Io) !void {
        var merged = try load(arena, io, std.fs.path.dirname(passes.path).?);
        for (passes.entries.keys(), passes.entries.values()) |hex, pass| {
            _ = merged.entries.orderedRemove(hex);
            try merged.entries.put(arena, hex, pass);
        }
        const keys = merged.entries.keys();
        const first = keys.len -| max_passes;
        var out: std.Io.Writer.Allocating = .init(arena);
        var w: json.Stringify = .{ .writer = &out.writer, .options = .{ .whitespace = .indent_1 } };
        try w.beginObject();
        try w.objectField("passes");
        try w.beginObject();
        for (keys[first..], merged.entries.values()[first..]) |hex, pass| {
            try w.objectField(hex);
            try w.write(pass);
        }
        try w.endObject();
        try w.endObject();
        try out.writer.writeByte('\n');
        try writeAtomic(io, passes.path, out.written());
    }
};

pub const TimesFile = struct {
    path: []const u8,
    times: select.Times = .empty,

    pub fn load(arena: Allocator, io: std.Io, dir: []const u8) Allocator.Error!TimesFile {
        var file: TimesFile = .{ .path = try std.fs.path.join(arena, &.{ dir, "times.json" }) };
        const doc = readJson(arena, io, file.path) orelse return file;
        if (doc != .object) return file;
        for (doc.object.keys(), doc.object.values()) |name, value| {
            const seconds: f64 = switch (value) {
                .float => |f| f,
                .integer => |i| @floatFromInt(i),
                else => continue,
            };
            try file.times.put(arena, name, seconds);
        }
        return file;
    }

    /// Merges `seconds` over what is on disk and writes, sorted by name.
    pub fn save(file: *const TimesFile, arena: Allocator, io: std.Io, seconds: []const struct { []const u8, f64 }) !void {
        var merged = try load(arena, io, std.fs.path.dirname(file.path).?);
        for (seconds) |entry| {
            try merged.times.put(arena, entry[0], @round(entry[1] * 10) / 10);
        }
        var names: std.ArrayList([]const u8) = .empty;
        var it = merged.times.keyIterator();
        while (it.next()) |name| try names.append(arena, name.*);
        std.mem.sort([]const u8, names.items, {}, struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lessThan);
        var out: std.Io.Writer.Allocating = .init(arena);
        var w: json.Stringify = .{ .writer = &out.writer, .options = .{ .whitespace = .indent_1 } };
        try w.beginObject();
        for (names.items) |name| {
            try w.objectField(name);
            try w.write(merged.times.get(name).?);
        }
        try w.endObject();
        try out.writer.writeByte('\n');
        try writeAtomic(io, file.path, out.written());
    }
};

const testing = std.testing;

const Scene = struct {
    state: std.heap.ArenaAllocator,
    m: manifest.Manifest,
    tree: tree_module.Tree,

    fn init() !Scene {
        var state: std.heap.ArenaAllocator = .init(testing.allocator);
        errdefer state.deinit();
        const a = state.allocator();
        const m = try manifest.load(a, try manifest.parse(a, manifest.sample));
        return .{ .state = state, .m = m, .tree = undefined };
    }

    fn scan(scene: *Scene, repository: tree_module.Memory) !void {
        const a = scene.state.allocator();
        scene.tree = try tree_module.build(a, testing.allocator, &scene.m, try repository.universe(a), repository);
    }

    fn deinit(scene: *Scene) void {
        scene.state.deinit();
    }
};

fn keyOf(scene: *Scene, gate: *const Gate, identity: []const u8) !Key {
    return key(scene.state.allocator(), &scene.m, &scene.tree, gate, identity);
}

/// The sample repository with one file's contents replaced.
fn edited(arena: Allocator, path: []const u8, contents: []const u8) !tree_module.Memory {
    const files = try arena.dupe(@typeInfo(@TypeOf(tree_module.sample_repository.files)).pointer.child, tree_module.sample_repository.files);
    for (files) |*file| {
        if (std.mem.eql(u8, file[0], path)) file[1] = contents;
    }
    return .{ .files = files };
}

test "the key ignores other families' files and comments, and follows shared code" {
    var scene: Scene = try .init();
    defer scene.deinit();
    const a = scene.state.allocator();
    const gate_a = &scene.m.gates[0];
    try scene.scan(tree_module.sample_repository);
    const base = try keyOf(&scene, gate_a, "m");

    try scene.scan(try edited(a, "inference/src/models/b.zig", "const s = @import(\"../shared.zig\");\nconst changed = 1;\n"));
    try testing.expectEqualStrings(&base, &try keyOf(&scene, gate_a, "m"));

    try scene.scan(try edited(a, "inference/src/shared.zig", "const std = @import(\"std\"); // why\n"));
    try testing.expectEqualStrings(&base, &try keyOf(&scene, gate_a, "m"));

    try scene.scan(try edited(a, "inference/src/shared.zig", "const std = @import(\"std\");\nconst x = 1;\n"));
    try testing.expect(!std.mem.eql(u8, &base, &try keyOf(&scene, gate_a, "m")));

    try scene.scan(try edited(a, "inference/src/root.zig", "const added = 1;\n"));
    try testing.expect(!std.mem.eql(u8, &base, &try keyOf(&scene, gate_a, "m")));
}

test "the key follows the model, the definition, and the optimize mode, not the prose" {
    var scene: Scene = try .init();
    defer scene.deinit();
    const a = scene.state.allocator();
    try scene.scan(tree_module.sample_repository);
    const gate_a = &scene.m.gates[0];
    const base = try keyOf(&scene, gate_a, "m");
    try testing.expect(!std.mem.eql(u8, &base, &try keyOf(&scene, gate_a, "other")));

    var gate = gate_a.*;
    var raw = try gate.raw.object.clone(a);
    try raw.put(a, "note", .{ .string = "reworded" });
    try raw.put(a, "evidence", .{ .string = "notes/elsewhere.md" });
    gate.raw = .{ .object = raw };
    try testing.expectEqualStrings(&base, &try keyOf(&scene, &gate, "m"));
    try raw.put(a, "tier", .{ .string = "verify-long" });
    gate.raw = .{ .object = raw };
    try testing.expect(!std.mem.eql(u8, &base, &try keyOf(&scene, &gate, "m")));

    scene.m.metal_optimize = "ReleaseFast";
    try testing.expect(!std.mem.eql(u8, &base, &try keyOf(&scene, gate_a, "m")));
}

test "the canonical form sorts keys and drops only the prose" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    const left = try manifest.parse(a, "{\"b\":[1,{\"z\":2,\"a\":\"x\"}],\"a\":true,\"note\":\"n\"}");
    const right = try manifest.parse(a, "{\"a\":true,\"b\":[1,{\"a\":\"x\",\"z\":2}]}");
    var gate = select_stub(left);
    const first = try canonicalGate(a, &gate);
    gate = select_stub(right);
    try testing.expectEqualStrings(first, try canonicalGate(a, &gate));
    try testing.expectEqualStrings("{\"a\":true,\"b\":[1,{\"a\":\"x\",\"z\":2}]}", first);
}

fn select_stub(raw: json.Value) Gate {
    return .{
        .name = "",
        .family = "",
        .tier = "",
        .model = "",
        .mtp = null,
        .baseline = null,
        .data = &.{},
        .command = &.{},
        .comparator = .exit,
        .evidence = "",
        .raw = raw,
    };
}

test "passes and times survive a round trip, merged with what is on disk" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try a.print(".zig-cache/tmp/{s}/gates", .{tmp.sub_path});

    var first = try Passes.load(a, testing.io, dir);
    try testing.expect(first.get("k1") == null);
    try first.put(a, "k1", .{ .gate = "g1", .revision = "abc", .seconds = 1.5 });
    try first.save(a, testing.io);

    var other = try Passes.load(a, testing.io, dir);
    try other.put(a, "k2", .{ .gate = "g2", .revision = "def", .seconds = 2 });
    try other.save(a, testing.io);

    try first.put(a, "k3", .{ .gate = "g3", .revision = "ghi", .seconds = 3 });
    try first.save(a, testing.io);

    const final = try Passes.load(a, testing.io, dir);
    try testing.expectEqualStrings("g1", final.get("k1").?.gate);
    try testing.expectEqualStrings("def", final.get("k2").?.revision);
    try testing.expectEqual(@as(f64, 3), final.get("k3").?.seconds);

    var times = try TimesFile.load(a, testing.io, dir);
    try times.save(a, testing.io, &.{.{ "g1", 1.26 }});
    times = try TimesFile.load(a, testing.io, dir);
    try times.save(a, testing.io, &.{.{ "g2", 7 }});
    times = try TimesFile.load(a, testing.io, dir);
    try testing.expectEqual(@as(f64, 1.3), times.times.get("g1").?);
    try testing.expectEqual(@as(f64, 7), times.times.get("g2").?);
}

test "a corrupt store reads as empty" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "passes.json", .data = "{not json" });
    const dir = try a.print(".zig-cache/tmp/{s}", .{tmp.sub_path});
    const passes = try Passes.load(a, testing.io, dir);
    try testing.expectEqual(@as(usize, 0), passes.entries.count());
}

test "the store follows NUCLIS_HOME, which must be absolute" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    var env: std.process.Environ.Map = .init(a);
    try env.put("HOME", "/h");
    try testing.expectEqualStrings("/h/.nuclis/gates", try directory(a, &env));
    try env.put("NUCLIS_HOME", "/data/n");
    try testing.expectEqualStrings("/data/n/gates", try directory(a, &env));
    try env.put("NUCLIS_HOME", "relative");
    try testing.expectError(error.InvalidHome, directory(a, &env));
}

test "a model's identity covers a directory's files and changes with their size" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "one.bin", .data = "aa" });
    const path = try a.print(".zig-cache/tmp/{s}", .{tmp.sub_path});
    const before = try modelIdentity(a, testing.io, &.{path});
    try testing.expect(std.mem.indexOf(u8, before, "one.bin 2 ") != null);
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "one.bin", .data = "aaa" });
    try testing.expect(!std.mem.eql(u8, before, try modelIdentity(a, testing.io, &.{path})));
}
