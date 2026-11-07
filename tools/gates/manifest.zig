//! `gates.json`: the registry of gates, models, data, checks, and the impact
//! rules. `validate` names every problem in the raw document; `load` turns a
//! valid one into typed records. Everything lives in the caller's arena.
//! The Python docs and workload scripts read the same file. Bounds live in
//! the manifest and nowhere else; docs/development.md § Gates.
const std = @import("std");
const json = std.json;
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub const tiers = [_][]const u8{ "verify", "verify-release", "verify-long", "verify-cpu" };
pub const placeholders = [_][]const u8{
    "{nuclis}", "{nuclis-cpu}", "{zig-metal}", "{zig-cpu}", "{model}",
    "{mtp}",    "{baseline}",   "{mmproj}",    "{trace}",
};
/// The family masks are one `u64`.
pub const max_families = 64;
/// More than any manifest needs; refuses a file that is not one.
pub const max_manifest_bytes = 16 << 20;

pub const Model = struct {
    key: []const u8,
    path: []const u8,
    mmproj: ?[]const u8,
};

pub const DataSet = struct {
    key: []const u8,
    fetch: []const u8,
    paths: []const []const u8,
    /// A set listed in a clip manifest: `root` and `listed_in`.
    root: ?[]const u8,
    listed_in: ?[]const u8,
};

pub const Trace = struct {
    reference: []const u8,
    positions: u32,
    embedding: ?u32,
    layers: ?u32,
    vocab: ?u32,
    draft: bool,
    max_absolute: f64,
    max_relative_rms: f64,
};

pub const Comparator = union(enum) {
    exit,
    trace: Trace,
};

pub const Gate = struct {
    name: []const u8,
    family: []const u8,
    tier: []const u8,
    model: []const u8,
    mtp: ?[]const u8,
    baseline: ?[]const u8,
    data: []const []const u8,
    command: []const []const u8,
    comparator: Comparator,
    evidence: []const u8,
    /// The manifest's object: what a cached pass keys on.
    raw: Value,
};

pub const Check = struct {
    name: []const u8,
    command: []const []const u8,
    paths: []const []const u8,
};

pub const Rule = struct {
    tier: []const u8,
    paths: []const []const u8,
    reason: []const u8,
};

pub const Family = struct {
    name: []const u8,
    seeds: []const []const u8,
};

pub const Manifest = struct {
    metal_optimize: []const u8,
    cpu_optimize: []const u8,
    cache: []const u8,
    models: []const Model,
    data: []const DataSet,
    checks: []const Check,
    requires: []const Rule,
    families: []const Family,
    aggregators: []const []const u8,
    family_cutoff: u32,
    canaries: []const []const u8,
    gates: []const Gate,

    pub fn model(m: *const Manifest, key: []const u8) ?*const Model {
        for (m.models) |*entry| {
            if (std.mem.eql(u8, entry.key, key)) return entry;
        }
        return null;
    }

    pub fn dataSet(m: *const Manifest, key: []const u8) ?*const DataSet {
        for (m.data) |*entry| {
            if (std.mem.eql(u8, entry.key, key)) return entry;
        }
        return null;
    }

    pub fn familyIndex(m: *const Manifest, name: []const u8) ?u6 {
        for (m.families, 0..) |family, i| {
            if (std.mem.eql(u8, family.name, name)) return @intCast(i);
        }
        return null;
    }

    pub fn gate(m: *const Manifest, name: []const u8) ?*const Gate {
        for (m.gates) |*entry| {
            if (std.mem.eql(u8, entry.name, name)) return entry;
        }
        return null;
    }
};

pub const ParseError = Allocator.Error || error{InvalidManifest};

pub fn parse(arena: Allocator, text: []const u8) ParseError!Value {
    return json.parseFromSliceLeaky(Value, arena, text, .{}) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidManifest,
    };
}

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

fn intField(v: Value, name: []const u8) ?i64 {
    const inner = field(v, name) orelse return null;
    return switch (inner) {
        .integer => |i| i,
        else => null,
    };
}

fn number(v: Value) ?f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
}

fn isStringList(v: ?Value) bool {
    const list = v orelse return false;
    if (list != .array or list.array.items.len == 0) return false;
    for (list.array.items) |item| {
        if (item != .string) return false;
    }
    return true;
}

fn isTier(name: ?[]const u8) bool {
    const text = name orelse return false;
    for (tiers) |tier| {
        if (std.mem.eql(u8, tier, text)) return true;
    }
    return false;
}

fn objectHas(v: Value, key: []const u8) bool {
    return field(v, key) != null;
}

fn commandUses(command: []const Value, token: []const u8) bool {
    for (command) |arg| {
        if (std.mem.indexOf(u8, arg.string, token) != null) return true;
    }
    return false;
}

const Problems = struct {
    arena: Allocator,
    list: std.ArrayList([]const u8) = .empty,

    fn add(p: *Problems, comptime format: []const u8, args: anytype) Allocator.Error!void {
        try p.list.append(p.arena, try p.arena.print(format, args));
    }
};

/// Every problem in the raw document; empty means `load` will succeed.
pub fn validate(arena: Allocator, doc: Value) Allocator.Error![]const []const u8 {
    var p: Problems = .{ .arena = arena };
    const empty_object: Value = .{ .object = .empty };
    const build = field(doc, "build") orelse empty_object;
    for ([_][]const u8{ "metal_optimize", "cpu_optimize", "cache" }) |key| {
        if (stringField(build, key) == null) try p.add("build.{s}: missing or not a string", .{key});
    }
    const models = field(doc, "models") orelse empty_object;
    if (models == .object) {
        for (models.object.keys(), models.object.values()) |name, entry| {
            if (stringField(entry, "path") == null) try p.add("models.{s}: needs a \"path\" string", .{name});
        }
    }
    const data = field(doc, "data") orelse empty_object;
    if (data == .object) {
        for (data.object.keys(), data.object.values()) |name, entry| {
            if (stringField(entry, "fetch") == null) {
                try p.add("data.{s}: needs a \"fetch\" command string", .{name});
            } else if (!isStringList(field(entry, "paths")) and
                (stringField(entry, "root") == null or stringField(entry, "listed_in") == null))
            {
                try p.add("data.{s}: needs \"paths\", or \"root\" and \"listed_in\"", .{name});
            }
        }
    }
    try validateImpact(&p, doc);
    const gates = field(doc, "gates") orelse Value{ .array = .init(arena) };
    if (gates == .array) {
        var seen: std.StringHashMapUnmanaged(void) = .empty;
        for (gates.array.items, 0..) |gate, i| {
            const name = stringField(gate, "name") orelse blk: {
                try p.add("gate #{d}: missing name", .{i});
                break :blk try arena.print("#{d}", .{i});
            };
            if ((try seen.getOrPut(arena, name)).found_existing) try p.add("{s}: duplicate name", .{name});
            try validateGate(&p, gate, name, models, data, doc);
        }
    }
    if (field(doc, "checks")) |checks| if (checks == .array) {
        for (checks.array.items, 0..) |check, i| {
            const name = stringField(check, "name") orelse try arena.print("check #{d}", .{i});
            if (!isStringList(field(check, "command"))) try p.add("{s}: command must be a non-empty argv list", .{name});
            if (!isStringList(field(check, "paths"))) try p.add("{s}: paths must be a non-empty list of globs", .{name});
        }
    };
    if (field(doc, "requires")) |rules| if (rules == .array) {
        for (rules.array.items, 0..) |rule, i| {
            const tier = stringField(rule, "tier");
            if (!isTier(tier) or std.mem.eql(u8, tier.?, "verify")) {
                try p.add("requires #{d}: tier must be one of verify-release, verify-long, verify-cpu", .{i});
            }
            if (!isStringList(field(rule, "paths"))) try p.add("requires #{d}: paths must be a non-empty list of globs", .{i});
            if (stringField(rule, "reason") == null) try p.add("requires #{d}: missing reason", .{i});
        }
    };
    return p.list.items;
}

fn validateImpact(p: *Problems, doc: Value) Allocator.Error!void {
    const families = field(doc, "families");
    if (families == null or families.? != .object or families.?.object.count() == 0) {
        try p.add("families: must map each family to its seed globs", .{});
    } else {
        if (families.?.object.count() > max_families) try p.add("families: more than {d}", .{max_families});
        for (families.?.object.keys(), families.?.object.values()) |name, seeds| {
            if (!isStringList(seeds)) try p.add("families.{s}: seeds must be a non-empty list of globs", .{name});
        }
    }
    if (!isStringList(field(doc, "aggregators"))) try p.add("aggregators: must be a non-empty list of paths", .{});
    const cutoff = intField(doc, "family_cutoff");
    if (cutoff == null or cutoff.? < 1) try p.add("family_cutoff: must be a positive integer", .{});
    if (!isStringList(field(doc, "canaries"))) {
        try p.add("canaries: must be a non-empty list of gate names", .{});
        return;
    }
    const gates = field(doc, "gates") orelse return;
    if (gates != .array) return;
    for (field(doc, "canaries").?.array.items) |canary| {
        for (gates.array.items) |gate| {
            if (!std.mem.eql(u8, stringField(gate, "name") orelse "", canary.string)) continue;
            if (!std.mem.eql(u8, stringField(gate, "tier") orelse "", "verify")) {
                try p.add("canaries: {s} is not in the verify tier", .{canary.string});
            }
            break;
        } else try p.add("canaries: no gate named {s}", .{canary.string});
    }
}

fn validateGate(p: *Problems, gate: Value, name: []const u8, models: Value, data: Value, doc: Value) Allocator.Error!void {
    if (!isTier(stringField(gate, "tier"))) {
        try p.add("{s}: tier must be one of verify, verify-release, verify-long, verify-cpu", .{name});
    }
    for ([_][]const u8{ "model", "mtp", "baseline" }) |key| {
        const value = stringField(gate, key) orelse continue;
        if (!objectHas(models, value)) try p.add("{s}: {s} names an unknown model \"{s}\"", .{ name, key, value });
    }
    if (!objectHas(gate, "model")) try p.add("{s}: missing model", .{name});
    const family = stringField(gate, "family");
    if (family == null or !objectHas(field(doc, "families") orelse Value.null, family.?)) {
        try p.add("{s}: family must be a key of families", .{name});
    }
    if (objectHas(gate, "paths")) try p.add("{s}: a gate carries no paths; impact comes from families", .{name});
    if (field(gate, "data")) |sets| if (sets == .array) {
        for (sets.array.items) |set| {
            if (set != .string or !objectHas(data, set.string)) {
                try p.add("{s}: data names an unknown set \"{s}\"", .{ name, if (set == .string) set.string else "?" });
            }
        }
    };
    const command = field(gate, "command");
    const valid_command = isStringList(command);
    if (!valid_command) {
        try p.add("{s}: command must be a non-empty argv list", .{name});
    } else {
        const argv = command.?.array.items;
        for (argv) |arg| try checkPlaceholders(p, name, arg.string);
        if (commandUses(argv, "{mtp}") and !objectHas(gate, "mtp")) {
            try p.add("{s}: command uses {{mtp}} but the gate names no mtp", .{name});
        }
        if (commandUses(argv, "{baseline}") and !objectHas(gate, "baseline")) {
            try p.add("{s}: command uses {{baseline}} but the gate names no baseline", .{name});
        }
        if (commandUses(argv, "{mmproj}")) {
            const model_key = stringField(gate, "model") orelse "";
            const entry = field(models, model_key) orelse Value.null;
            if (!objectHas(entry, "mmproj")) {
                try p.add("{s}: command uses {{mmproj}} but model \"{s}\" names no mmproj", .{ name, model_key });
            }
        }
    }
    const comparator = field(gate, "comparator") orelse Value.null;
    const kind = stringField(comparator, "kind") orelse "";
    if (std.mem.eql(u8, kind, "trace")) {
        if (stringField(comparator, "reference") == null) try p.add("{s}: trace comparator needs a reference directory", .{name});
        if (intField(comparator, "positions") == null) try p.add("{s}: trace comparator needs integer positions", .{name});
        const bounds = field(gate, "bounds");
        const well_formed = bounds != null and bounds.? == .object and bounds.?.object.count() == 2 and
            number(field(bounds.?, "max_absolute") orelse Value.null) != null and
            number(field(bounds.?, "max_relative_rms") orelse Value.null) != null;
        if (!well_formed) try p.add("{s}: trace gates need bounds {{max_absolute, max_relative_rms}}", .{name});
        if (valid_command and !commandUses(command.?.array.items, "{trace}")) {
            try p.add("{s}: trace gates must write to {{trace}}", .{name});
        }
    } else if (std.mem.eql(u8, kind, "exit")) {
        if (objectHas(gate, "bounds")) try p.add("{s}: an exit gate carries no bounds", .{name});
    } else {
        try p.add("{s}: comparator.kind must be one of exit, trace", .{name});
        if (objectHas(gate, "bounds")) try p.add("{s}: an exit gate carries no bounds", .{name});
    }
    if (stringField(gate, "evidence") == null) try p.add("{s}: missing evidence", .{name});
}

/// Each `{lowercase-word}` in `arg` must be a known placeholder.
fn checkPlaceholders(p: *Problems, name: []const u8, arg: []const u8) Allocator.Error!void {
    var i: usize = 0;
    while (std.mem.indexOfScalarPos(u8, arg, i, '{')) |open| {
        var end = open + 1;
        while (end < arg.len and (std.ascii.isLower(arg[end]) or arg[end] == '-')) end += 1;
        i = open + 1;
        if (end == open + 1 or end == arg.len or arg[end] != '}') continue;
        const token = arg[open .. end + 1];
        for (placeholders) |known| {
            if (std.mem.eql(u8, known, token)) break;
        } else try p.add("{s}: unknown placeholder {s}", .{ name, token });
    }
}

fn strings(arena: Allocator, v: ?Value) Allocator.Error![]const []const u8 {
    const list = v orelse return &.{};
    const out = try arena.alloc([]const u8, list.array.items.len);
    for (out, list.array.items) |*slot, item| slot.* = item.string;
    return out;
}

fn optionalU32(v: Value, name: []const u8) ?u32 {
    const value = intField(v, name) orelse return null;
    return std.math.cast(u32, value);
}

/// The typed manifest of a document `validate` accepted.
pub fn load(arena: Allocator, doc: Value) Allocator.Error!Manifest {
    const build = field(doc, "build").?;
    const models_object = field(doc, "models").?.object;
    const models = try arena.alloc(Model, models_object.count());
    for (models, models_object.keys(), models_object.values()) |*slot, key, entry| {
        slot.* = .{ .key = key, .path = stringField(entry, "path").?, .mmproj = stringField(entry, "mmproj") };
    }
    const data_object = if (field(doc, "data")) |d| d.object else json.ObjectMap.empty;
    const data = try arena.alloc(DataSet, data_object.count());
    for (data, data_object.keys(), data_object.values()) |*slot, key, entry| {
        slot.* = .{
            .key = key,
            .fetch = stringField(entry, "fetch").?,
            .paths = try strings(arena, field(entry, "paths")),
            .root = stringField(entry, "root"),
            .listed_in = stringField(entry, "listed_in"),
        };
    }
    const checks_value = if (field(doc, "checks")) |c| c.array.items else &[_]Value{};
    const checks = try arena.alloc(Check, checks_value.len);
    for (checks, checks_value) |*slot, check| {
        slot.* = .{
            .name = stringField(check, "name") orelse "check",
            .command = try strings(arena, field(check, "command")),
            .paths = try strings(arena, field(check, "paths")),
        };
    }
    const rules_value = if (field(doc, "requires")) |r| r.array.items else &[_]Value{};
    const rules = try arena.alloc(Rule, rules_value.len);
    for (rules, rules_value) |*slot, rule| {
        slot.* = .{
            .tier = stringField(rule, "tier").?,
            .paths = try strings(arena, field(rule, "paths")),
            .reason = stringField(rule, "reason").?,
        };
    }
    const families_object = field(doc, "families").?.object;
    const families = try arena.alloc(Family, families_object.count());
    for (families, families_object.keys(), families_object.values()) |*slot, name, seeds| {
        slot.* = .{ .name = name, .seeds = try strings(arena, seeds) };
    }
    const gates_value = field(doc, "gates").?.array.items;
    const gates = try arena.alloc(Gate, gates_value.len);
    for (gates, gates_value) |*slot, gate| {
        const comparator = field(gate, "comparator").?;
        slot.* = .{
            .name = stringField(gate, "name").?,
            .family = stringField(gate, "family").?,
            .tier = stringField(gate, "tier").?,
            .model = stringField(gate, "model").?,
            .mtp = stringField(gate, "mtp"),
            .baseline = stringField(gate, "baseline"),
            .data = try strings(arena, field(gate, "data")),
            .command = try strings(arena, field(gate, "command")),
            .comparator = if (std.mem.eql(u8, stringField(comparator, "kind").?, "exit")) .exit else .{ .trace = .{
                .reference = stringField(comparator, "reference").?,
                .positions = optionalU32(comparator, "positions") orelse 0,
                .embedding = optionalU32(comparator, "embedding"),
                .layers = optionalU32(comparator, "layers"),
                .vocab = optionalU32(comparator, "vocab"),
                .draft = if (field(comparator, "draft")) |d| d == .bool and d.bool else false,
                .max_absolute = number(field(field(gate, "bounds").?, "max_absolute").?).?,
                .max_relative_rms = number(field(field(gate, "bounds").?, "max_relative_rms").?).?,
            } },
            .evidence = stringField(gate, "evidence").?,
            .raw = gate,
        };
    }
    return .{
        .metal_optimize = stringField(build, "metal_optimize").?,
        .cpu_optimize = stringField(build, "cpu_optimize").?,
        .cache = stringField(build, "cache").?,
        .models = models,
        .data = data,
        .checks = checks,
        .requires = rules,
        .families = families,
        .aggregators = try strings(arena, field(doc, "aggregators")),
        .family_cutoff = @intCast(intField(doc, "family_cutoff").?),
        .canaries = try strings(arena, field(doc, "canaries")),
        .gates = gates,
    };
}

/// A document for the tests of this tool: three gates of one family.
pub const sample =
    \\{
    \\ "build": {"metal_optimize": "ReleaseSafe", "cpu_optimize": "ReleaseFast", "cache": ".zig-cache/global"},
    \\ "models": {"a": {"path": "~/m/a.gguf"}, "a_mtp": {"path": "~/m/a-mtp.gguf"}},
    \\ "data": {
    \\  "text": {"paths": ["eval/t.raw"], "fetch": "make eval-corpus"},
    \\  "clips": {"root": "speech", "listed_in": "corpus.json", "fetch": "make speech-corpus"}
    \\ },
    \\ "families": {"a": ["inference/src/models/a*.zig"], "b": ["inference/src/models/b*.zig"]},
    \\ "aggregators": ["inference/src/root.zig"],
    \\ "family_cutoff": 1,
    \\ "canaries": ["a-trace"],
    \\ "gates": [
    \\  {"name": "a-trace", "family": "a", "tier": "verify", "model": "a",
    \\   "command": ["{nuclis}", "generate", "--model", "{model}", "--trace-dir", "{trace}"],
    \\   "comparator": {"kind": "trace", "reference": "tests/fixtures/a", "positions": 2},
    \\   "bounds": {"max_absolute": 0.002, "max_relative_rms": 0.0001}, "evidence": "docs/a.md", "note": "n"},
    \\  {"name": "a-draft-cpu", "family": "a", "tier": "verify-cpu", "model": "a", "mtp": "a_mtp",
    \\   "command": ["{zig-cpu}", "test-generation", "--", "{model}", "--draft-model", "{mtp}"],
    \\   "comparator": {"kind": "exit"}, "evidence": "docs/a.md"},
    \\  {"name": "a-perplexity-4k", "family": "a", "tier": "verify-long", "model": "a",
    \\   "command": ["{nuclis}", "eval", "--model", "{model}", "--file", "t.raw", "--reference", "r.json"],
    \\   "comparator": {"kind": "exit"}, "evidence": "docs/a.md"},
    \\  {"name": "b-metal", "family": "b", "tier": "verify", "model": "a",
    \\   "command": ["{nuclis}", "--model", "{model}"],
    \\   "comparator": {"kind": "exit"}, "evidence": "docs/b.md"}
    \\ ],
    \\ "checks": [
    \\  {"name": "fmt", "command": ["zig", "fmt", "--check", "{zig-files}"], "paths": ["**/*.zig"]},
    \\  {"name": "metal", "command": ["make", "test-metal"], "paths": ["inference/src/backends/**"]}
    \\ ],
    \\ "requires": [
    \\  {"tier": "verify-cpu", "paths": ["inference/src/backends/cpu/**", "inference/src/models/*_runtime.zig"],
    \\   "reason": "the CPU reference"}
    \\ ]
    \\}
;

const testing = std.testing;

const Test = struct {
    arena: std.heap.ArenaAllocator,
    doc: Value,

    fn init() !Test {
        var arena: std.heap.ArenaAllocator = .init(testing.allocator);
        errdefer arena.deinit();
        const doc = try parse(arena.allocator(), sample);
        return .{ .arena = arena, .doc = doc };
    }

    fn deinit(t: *Test) void {
        t.arena.deinit();
    }

    fn gate(t: *Test, i: usize) *Value {
        return &t.doc.object.getPtr("gates").?.array.items[i];
    }

    fn set(t: *Test, target: *Value, key: []const u8, value: Value) !void {
        try target.object.put(t.arena.allocator(), key, value);
    }

    fn expectProblem(t: *Test, needle: []const u8) !void {
        const problems = try validate(t.arena.allocator(), t.doc);
        for (problems) |problem| {
            if (std.mem.indexOf(u8, problem, needle) != null) return;
        }
        std.debug.print("no problem names \"{s}\" in: {f}\n", .{ needle, json.fmt(problems, .{}) });
        return error.TestExpectedProblem;
    }
};

test "the sample is valid and loads" {
    var t: Test = try .init();
    defer t.deinit();
    const a = t.arena.allocator();
    try testing.expectEqual(@as(usize, 0), (try validate(a, t.doc)).len);
    const m = try load(a, t.doc);
    try testing.expectEqual(@as(usize, 4), m.gates.len);
    try testing.expectEqual(@as(u32, 2), m.gates[0].comparator.trace.positions);
    try testing.expectEqual(@as(f64, 0.002), m.gates[0].comparator.trace.max_absolute);
    try testing.expectEqualStrings("a_mtp", m.gates[1].mtp.?);
    try testing.expectEqual(@as(?u6, 1), m.familyIndex("b"));
    try testing.expectEqualStrings("speech", m.dataSet("clips").?.root.?);
}

test "validation names each problem" {
    var t: Test = try .init();
    defer t.deinit();
    const gates = t.doc.object.getPtr("gates").?;
    try gates.array.append(gates.array.items[0]);
    try t.set(t.gate(1), "tier", .{ .string = "cheap" });
    try t.set(t.gate(1), "model", .{ .string = "nope" });
    var bogus: json.Array = .init(t.arena.allocator());
    try bogus.append(.{ .string = "{bogus}" });
    try t.set(t.gate(0), "command", .{ .array = bogus });
    var comparator: json.ObjectMap = .empty;
    try comparator.put(t.arena.allocator(), "kind", .{ .string = "report" });
    try t.set(t.gate(0), "comparator", .{ .object = comparator });
    try t.expectProblem("duplicate name");
    try t.expectProblem("tier must be");
    try t.expectProblem("unknown model \"nope\"");
    try t.expectProblem("unknown placeholder {bogus}");
    try t.expectProblem("comparator.kind");
}

test "an exit gate refuses bounds and a trace gate needs them" {
    var t: Test = try .init();
    defer t.deinit();
    const bounds = t.gate(0).object.get("bounds").?;
    try t.set(t.gate(1), "bounds", bounds);
    _ = t.gate(0).object.orderedRemove("bounds");
    try t.expectProblem("carries no bounds");
    try t.expectProblem("need bounds");
}

test "validation names bad data sets" {
    var t: Test = try .init();
    defer t.deinit();
    const a = t.arena.allocator();
    const data = t.doc.object.getPtr("data").?;
    var bad: json.ObjectMap = .empty;
    var paths: json.Array = .init(a);
    try paths.append(.{ .string = "x" });
    try bad.put(a, "paths", .{ .array = paths });
    try data.object.put(a, "bad", .{ .object = bad });
    var loose: json.ObjectMap = .empty;
    try loose.put(a, "root", .{ .string = "r" });
    try loose.put(a, "fetch", .{ .string = "make x" });
    try data.object.put(a, "loose", .{ .object = loose });
    var nowhere: json.Array = .init(a);
    try nowhere.append(.{ .string = "nowhere" });
    try t.set(t.gate(0), "data", .{ .array = nowhere });
    try t.expectProblem("data.bad: needs a \"fetch\" command string");
    try t.expectProblem("data.loose: needs \"paths\", or \"root\" and \"listed_in\"");
    try t.expectProblem("a-trace: data names an unknown set \"nowhere\"");
}

test "validation names bad checks and rules" {
    var t: Test = try .init();
    defer t.deinit();
    const a = t.arena.allocator();
    const empty: json.Array = .init(a);
    var check: json.ObjectMap = .empty;
    try check.put(a, "name", .{ .string = "bad" });
    try check.put(a, "command", .{ .array = empty });
    try check.put(a, "paths", .{ .array = empty });
    try t.doc.object.getPtr("checks").?.array.append(.{ .object = check });
    var rule: json.ObjectMap = .empty;
    try rule.put(a, "tier", .{ .string = "verify" });
    var paths: json.Array = .init(a);
    try paths.append(.{ .string = "x" });
    try rule.put(a, "paths", .{ .array = paths });
    try rule.put(a, "reason", .{ .string = "r" });
    try t.doc.object.getPtr("requires").?.array.append(.{ .object = rule });
    try t.expectProblem("bad: command");
    try t.expectProblem("bad: paths");
    try t.expectProblem("requires #1: tier");
}

test "validation names impact problems" {
    var t: Test = try .init();
    defer t.deinit();
    const a = t.arena.allocator();
    try t.set(&t.doc, "family_cutoff", .{ .integer = 0 });
    var canaries: json.Array = .init(a);
    try canaries.append(.{ .string = "a-draft-cpu" });
    try canaries.append(.{ .string = "nothing" });
    try t.set(&t.doc, "canaries", .{ .array = canaries });
    try t.set(t.gate(0), "family", .{ .string = "z" });
    var stale: json.Array = .init(a);
    try stale.append(.{ .string = "inference/**" });
    try t.set(t.gate(1), "paths", .{ .array = stale });
    try t.expectProblem("family_cutoff");
    try t.expectProblem("a-draft-cpu is not in the verify tier");
    try t.expectProblem("no gate named nothing");
    try t.expectProblem("a-trace: family must be a key of families");
    try t.expectProblem("a-draft-cpu: a gate carries no paths");
}

test "placeholders the gate cannot fill are refused" {
    var t: Test = try .init();
    defer t.deinit();
    const a = t.arena.allocator();
    var argv: json.Array = .init(a);
    try argv.append(.{ .string = "{nuclis}" });
    try argv.append(.{ .string = "--a={mtp}" });
    try argv.append(.{ .string = "{baseline}" });
    try argv.append(.{ .string = "{mmproj}" });
    try t.set(t.gate(3), "command", .{ .array = argv });
    try t.expectProblem("b-metal: command uses {mtp}");
    try t.expectProblem("b-metal: command uses {baseline}");
    try t.expectProblem("b-metal: command uses {mmproj}");
}
