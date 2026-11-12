//! The engine tree: every file a gate's result can depend on, with its
//! fingerprint and the families that reach it.
//!
//! Inputs are the `inference/` sources, what they embed, what a family seeds,
//! and the fixtures a gate names. The forward graph comes from `@import("x.zig")`
//! and `@embedFile("y")` tokens; a `.metal` string literal counts as an embed,
//! because `kernels.zig` joins its `kernel_files` list in a loop. A family's
//! reach walks forward from its seeds and neither enters nor counts an
//! aggregator, so an aggregator belongs to no family: changing one reaches
//! everything, which is what no family means to `select`.
//!
//! Outside the engine, the tree keeps the context a remembered pass also
//! depends on: the build files, which every gate's build reads, and the
//! application (`src/`, `huggingface/`), which only a `{nuclis}` binary
//! contains. Context is never an engine change, so `select` ignores it.
const std = @import("std");
const glob = @import("glob.zig");
const fingerprint = @import("fingerprint.zig");
const manifest = @import("manifest.zig");
const Allocator = std.mem.Allocator;
const Manifest = manifest.Manifest;

/// A source past this is not one of ours.
pub const max_input_bytes = 256 << 20;

const source_extensions = [_][]const u8{ ".zig", ".metal", ".h", ".m" };

pub fn isSource(path: []const u8) bool {
    if (!std.mem.startsWith(u8, path, "inference/")) return false;
    for (source_extensions) |extension| {
        if (std.mem.endsWith(u8, path, extension)) return true;
    }
    return false;
}

/// What a context file reaches: every gate, or the gates that run the binary.
pub const Scope = enum { build, application };

/// The scope of a path outside the engine, or null for an engine path.
pub fn contextScope(path: []const u8) ?Scope {
    const build_files = [_][]const u8{ "build.zig", "build.zig.zon", "inference/build.zig.zon" };
    for (build_files) |name| {
        if (std.mem.eql(u8, path, name)) return .build;
    }
    if (std.mem.startsWith(u8, path, "src/") or std.mem.startsWith(u8, path, "huggingface/")) return .application;
    return null;
}

pub const Context = struct {
    path: []const u8,
    digest: fingerprint.Digest,
    scope: Scope,
};

pub const File = struct {
    path: []const u8,
    digest: fingerprint.Digest,
    /// Bit i set: family i reaches this file. Zero: no family owns it.
    families: u64 = 0,
    edges: []const []const u8 = &.{},
};

pub const Tree = struct {
    /// Sorted by path.
    files: []File,
    /// Sorted by path; never an engine input.
    context: []const Context = &.{},

    pub fn get(tree: *const Tree, path: []const u8) ?*const File {
        var low: usize = 0;
        var high = tree.files.len;
        while (low < high) {
            const middle = low + (high - low) / 2;
            switch (std.mem.order(u8, tree.files[middle].path, path)) {
                .lt => low = middle + 1,
                .gt => high = middle,
                .eq => return &tree.files[middle],
            }
        }
        return null;
    }

    /// The families of `path`; one the tree no longer holds answers by seed.
    pub fn familiesOf(tree: *const Tree, m: *const Manifest, path: []const u8) u64 {
        if (tree.get(path)) |file| return file.families;
        return seedFamilies(m, path);
    }
};

fn seedFamilies(m: *const Manifest, path: []const u8) u64 {
    var mask: u64 = 0;
    if (glob.matchesAny(m.aggregators, path)) return 0;
    for (m.families, 0..) |family, i| {
        if (glob.matchesAny(family.seeds, path)) mask |= @as(u64, 1) << @intCast(i);
    }
    return mask;
}

/// True when a change to `path` is an engine change: a tracked input, or a
/// file that would be one had it not been removed.
pub fn isInput(tree: *const Tree, m: *const Manifest, path: []const u8) bool {
    return tree.get(path) != null or isSource(path) or seedFamilies(m, path) != 0;
}

/// The targets a Zig source names, resolved to repository-relative paths.
pub fn edgesOf(arena: Allocator, path: []const u8, source: [:0]const u8) Allocator.Error![]const []const u8 {
    const directory = std.fs.path.dirnamePosix(path) orelse ".";
    var out: std.ArrayList([]const u8) = .empty;
    var tokens = std.zig.Tokenizer.init(source);
    const State = enum { idle, import, embed, import_paren, embed_paren };
    var state: State = .idle;
    while (true) {
        const token = tokens.next();
        if (token.tag == .eof) break;
        const text = source[token.loc.start..token.loc.end];
        var target: ?[]const u8 = null;
        if (token.tag == .string_literal and text.len >= 2) {
            const literal = text[1 .. text.len - 1];
            if (std.mem.endsWith(u8, literal, ".metal")) target = literal;
            if (state == .embed_paren) target = literal;
            if (state == .import_paren and std.mem.endsWith(u8, literal, ".zig")) target = literal;
        }
        state = switch (token.tag) {
            .builtin => if (std.mem.eql(u8, text, "@import"))
                .import
            else if (std.mem.eql(u8, text, "@embedFile"))
                .embed
            else
                .idle,
            .l_paren => switch (state) {
                .import => .import_paren,
                .embed => .embed_paren,
                else => .idle,
            },
            else => .idle,
        };
        const named = target orelse continue;
        try out.append(arena, try std.fs.path.resolveAllocPosix(arena, &.{ directory, named }));
    }
    return out.items;
}

const gate_scripts = "scripts/gate";

/// The files of `universe` a gate names in its arguments, its reference, or a
/// data set's listing: exact paths, or everything under a named directory.
fn namedBy(arena: Allocator, m: *const Manifest, gate: *const manifest.Gate, universe: []const []const u8) Allocator.Error![]const []const u8 {
    var names: std.ArrayList([]const u8) = .empty;
    try names.appendSlice(arena, gate.command);
    if (gate.comparator == .trace) try names.append(arena, gate.comparator.trace.reference);
    for (gate.data) |key| {
        const set = m.dataSet(key) orelse continue;
        if (set.listed_in) |listing| try names.append(arena, listing);
    }
    var out: std.ArrayList([]const u8) = .empty;
    for (names.items) |argument| {
        // A gate script takes its siblings, which import one another.
        const name = if (std.mem.startsWith(u8, argument, gate_scripts ++ "/")) gate_scripts else argument;
        if (name.len == 0 or name[0] == '-' or name[0] == '{') continue;
        for (universe) |path| {
            if (contextScope(path) != null or !std.mem.startsWith(u8, path, name)) continue;
            if (path.len == name.len or path[name.len] == '/') try out.append(arena, path);
        }
    }
    return out.items;
}

/// Builds the tree from `universe` (sorted paths under `inference/`,
/// `tests/fixtures/`, `scripts/gate/`, and the context) through
/// `reader.read(path)`, whose bytes need last only until the next read.
/// `arena` owns the result.
pub fn build(arena: Allocator, gpa: Allocator, m: *const Manifest, universe: []const []const u8, reader: anytype) !Tree {
    var known: std.StringHashMapUnmanaged(void) = .empty;
    for (universe) |path| try known.put(arena, path, {});

    var queue: std.ArrayList([]const u8) = .empty;
    var seen: std.StringHashMapUnmanaged(u32) = .empty;
    var files: std.ArrayList(File) = .empty;
    const enqueue = struct {
        fn call(a: Allocator, q: *std.ArrayList([]const u8), s: *std.StringHashMapUnmanaged(u32), path: []const u8) Allocator.Error!void {
            const entry = try s.getOrPut(a, path);
            if (entry.found_existing) return;
            entry.value_ptr.* = @intCast(q.items.len);
            try q.append(a, path);
        }
    }.call;

    var context: std.ArrayList(Context) = .empty;
    for (universe) |path| {
        if (contextScope(path)) |scope| {
            try context.append(arena, .{ .path = path, .digest = try fingerprint.ofFile(gpa, path, try reader.read(path)), .scope = scope });
            continue;
        }
        if (isSource(path) or seedFamilies(m, path) != 0) try enqueue(arena, &queue, &seen, path);
    }
    var named: std.StringHashMapUnmanaged(u64) = .empty;
    for (m.gates) |*gate| {
        const bit = @as(u64, 1) << (m.familyIndex(gate.family) orelse continue);
        for (try namedBy(arena, m, gate, universe)) |path| {
            try enqueue(arena, &queue, &seen, path);
            const entry = try named.getOrPut(arena, path);
            if (!entry.found_existing) entry.value_ptr.* = 0;
            entry.value_ptr.* |= bit;
        }
    }

    var next: usize = 0;
    while (next < queue.items.len) : (next += 1) {
        const path = queue.items[next];
        const bytes = try reader.read(path);
        var edges: []const []const u8 = &.{};
        if (std.mem.endsWith(u8, path, ".zig")) {
            const source = try gpa.dupeSentinel(u8, bytes, 0);
            defer gpa.free(source);
            var kept: std.ArrayList([]const u8) = .empty;
            for (try edgesOf(arena, path, source)) |target| {
                if (!known.contains(target)) continue;
                try kept.append(arena, target);
                try enqueue(arena, &queue, &seen, target);
            }
            edges = kept.items;
        }
        try files.append(arena, .{ .path = path, .digest = try fingerprint.ofFile(gpa, path, bytes), .edges = edges });
    }

    // `seen` maps a path to its queue position, which is also its `files` index.
    for (m.families, 0..) |family, i| {
        const bit = @as(u64, 1) << @intCast(i);
        var stack: std.ArrayList([]const u8) = .empty;
        for (universe) |path| {
            if (glob.matchesAny(family.seeds, path)) try stack.append(arena, path);
        }
        var visited: std.StringHashMapUnmanaged(void) = .empty;
        while (stack.pop()) |path| {
            if (glob.matchesAny(m.aggregators, path)) continue;
            if ((try visited.getOrPut(arena, path)).found_existing) continue;
            const file = &files.items[seen.get(path).?];
            file.families |= bit;
            try stack.appendSlice(arena, file.edges);
        }
    }
    var it = named.iterator();
    while (it.next()) |entry| files.items[seen.get(entry.key_ptr.*).?].families |= entry.value_ptr.*;

    std.mem.sort(File, files.items, {}, struct {
        fn lessThan(_: void, a: File, b: File) bool {
            return std.mem.lessThan(u8, a.path, b.path);
        }
    }.lessThan);
    return .{ .files = files.items, .context = context.items };
}

/// Problems only the real tree can show: a seed or an aggregator that matches
/// nothing, a kernel no source reaches.
pub fn lint(arena: Allocator, m: *const Manifest, tree: *const Tree, universe: []const []const u8) Allocator.Error![]const []const u8 {
    var problems: std.ArrayList([]const u8) = .empty;
    for (m.families) |family| {
        for (family.seeds) |seed| {
            for (universe) |path| {
                if (glob.matches(seed, path)) break;
            } else try problems.append(arena, try arena.print("families.{s}: {s} matches no file", .{ family.name, seed }));
        }
    }
    for (m.aggregators) |aggregator| {
        for (universe) |path| {
            if (glob.matches(aggregator, path)) break;
        } else try problems.append(arena, try arena.print("aggregators: {s} matches no file", .{aggregator}));
    }
    var targets: std.StringHashMapUnmanaged(void) = .empty;
    for (tree.files) |file| {
        for (file.edges) |edge| try targets.put(arena, edge, {});
    }
    for (universe) |path| {
        if (!std.mem.endsWith(u8, path, ".metal") or targets.contains(path)) continue;
        try problems.append(arena, try arena.print("{s}: no source reaches this kernel", .{path}));
    }
    // A retired model's fixtures go with it.
    for (m.retired) |prefix| {
        const root = try std.mem.concat(arena, u8, &.{ "tests/fixtures/", prefix });
        for (universe) |path| {
            if (std.mem.startsWith(u8, path, root)) try problems.append(arena, try arena.print("{s}: a fixture of the retired \"{s}\"; delete it", .{ path, prefix }));
        }
    }
    return problems.items;
}

const testing = std.testing;

/// An in-memory repository for the tests.
pub const Memory = struct {
    files: []const struct { []const u8, []const u8 },

    pub fn read(memory: Memory, path: []const u8) error{FileNotFound}![]const u8 {
        for (memory.files) |file| {
            if (std.mem.eql(u8, file[0], path)) return file[1];
        }
        return error.FileNotFound;
    }

    pub fn universe(memory: Memory, arena: Allocator) Allocator.Error![]const []const u8 {
        const paths = try arena.alloc([]const u8, memory.files.len);
        for (paths, memory.files) |*slot, file| slot.* = file[0];
        std.mem.sort([]const u8, paths, {}, struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lessThan);
        return paths;
    }
};

pub const sample_repository: Memory = .{ .files = &.{
    .{ "inference/src/root.zig", "pub const a = @import(\"models/a.zig\");\npub const b = @import(\"models/b.zig\");\n" },
    .{ "inference/src/models/a.zig", "const s = @import(\"../shared.zig\");\nconst r = @import(\"../root.zig\");\nconst j = @embedFile(\"fixtures/a.json\");\n" },
    .{ "inference/src/models/a_metal.zig", "const k = @import(\"../kernels.zig\");\n" },
    .{ "inference/src/models/b.zig", "const s = @import(\"../shared.zig\");\nconst k = @import(\"../kernels.zig\");\n" },
    .{ "inference/src/models/fixtures/a.json", "{}" },
    .{ "inference/src/shared.zig", "const std = @import(\"std\");\n" },
    .{ "inference/src/kernels.zig", "const files = [_][]const u8{ \"kernels/one.metal\", \"kernels/two.metal\" };\nfn f() void { for (files) |file| _ = @embedFile(file); }\n" },
    .{ "inference/src/kernels/one.metal", "kernel one\n" },
    .{ "inference/src/kernels/two.metal", "kernel two\n" },
    .{ "inference/src/lonely.zig", "// nothing imports this\n" },
    .{ "tests/fixtures/a/x.f32", "abcd" },
    .{ "tests/fixtures/unused/y.f32", "abcd" },
    .{ "scripts/gate/check.py", "import helper\n" },
    .{ "scripts/gate/helper.py", "x = 1\n" },
    .{ "src/main.zig", "pub fn main() void {}\n" },
    .{ "build.zig", "const std = @import(\"std\");\n" },
} };

fn sampleManifest(arena: Allocator) !Manifest {
    return manifest.load(arena, try manifest.parse(arena, manifest.sample));
}

test "edges cover imports, embeds, and a kernel list, and resolve relative paths" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    const source =
        \\const std = @import("std");
        \\const inference = @import("inference");
        \\const up = @import("../quant/q.zig");
        \\// @import("commented.zig");
        \\const text = @embedFile("fixtures/a.json");
        \\const list = [_][]const u8{ "kernels/one.metal", "not-a-path" };
        \\const not = "@import(\"x.zig\")";
    ;
    const edges = try edgesOf(a, "inference/src/models/a.zig", source);
    try testing.expectEqual(@as(usize, 3), edges.len);
    try testing.expectEqualStrings("inference/src/quant/q.zig", edges[0]);
    try testing.expectEqualStrings("inference/src/models/fixtures/a.json", edges[1]);
    try testing.expectEqualStrings("inference/src/models/kernels/one.metal", edges[2]);
}

test "a family's reach stops at an aggregator and an aggregator has no family" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    const m = try sampleManifest(a);
    const universe = try sample_repository.universe(a);
    const tree = try build(a, testing.allocator, &m, universe, sample_repository);
    const family_a: u64 = 1;
    const family_b: u64 = 2;
    try testing.expectEqual(family_a, tree.get("inference/src/models/a.zig").?.families);
    try testing.expectEqual(family_a, tree.get("inference/src/models/fixtures/a.json").?.families);
    try testing.expectEqual(family_a | family_b, tree.get("inference/src/shared.zig").?.families);
    try testing.expectEqual(family_a | family_b, tree.get("inference/src/kernels/two.metal").?.families);
    try testing.expectEqual(@as(u64, 0), tree.get("inference/src/root.zig").?.families);
    try testing.expectEqual(@as(u64, 0), tree.get("inference/src/lonely.zig").?.families);
    // A fixture a gate names joins the tree with that gate's family; others stay out.
    try testing.expectEqual(family_a, tree.get("tests/fixtures/a/x.f32").?.families);
    try testing.expect(tree.get("tests/fixtures/unused/y.f32") == null);
}

test "lint names a dead seed, a dead aggregator, and an unreached kernel" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    var m = try sampleManifest(a);
    const universe = try sample_repository.universe(a);
    const tree = try build(a, testing.allocator, &m, universe, sample_repository);
    try testing.expectEqual(@as(usize, 0), (try lint(a, &m, &tree, universe)).len);

    m.aggregators = &.{"inference/src/gone.zig"};
    const families = [_]manifest.Family{.{ .name = "a", .seeds = &.{"inference/src/gone/*.zig"} }};
    m.families = &families;
    const extended = try a.dupe([]const u8, universe);
    const orphan = try std.mem.concat(a, []const u8, &.{ extended, &.{"inference/src/orphan.metal"} });
    const problems = try lint(a, &m, &tree, orphan);
    try testing.expectEqual(@as(usize, 3), problems.len);

    var clean = try sampleManifest(a);
    clean.retired = &.{"unused"};
    const retired = try lint(a, &clean, &tree, universe);
    try testing.expectEqual(@as(usize, 1), retired.len);
    try testing.expect(std.mem.indexOf(u8, retired[0], "tests/fixtures/unused/y.f32: a fixture of the retired") != null);
}

test "a removed file answers by its seeds" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    const m = try sampleManifest(a);
    const universe = try sample_repository.universe(a);
    const tree = try build(a, testing.allocator, &m, universe, sample_repository);
    try testing.expectEqual(@as(u64, 2), tree.familiesOf(&m, "inference/src/models/b_gone.zig"));
    try testing.expectEqual(@as(u64, 0), tree.familiesOf(&m, "inference/src/gone.zig"));
    try testing.expect(isInput(&tree, &m, "inference/src/gone.zig"));
    try testing.expect(!isInput(&tree, &m, "src/main.zig"));
    try testing.expect(!isInput(&tree, &m, "docs/x.md"));
}
