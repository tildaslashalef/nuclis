//! What a change selects. A changed input reached by exactly the families in
//! S selects S's gates of the tier; when S is empty or past `family_cutoff`
//! the change is shared, and selects the canaries (another tier has none, so
//! there it selects the whole tier). Checks and `requires` rules stay path
//! globs. `byCost` is the run order.
const std = @import("std");
const glob = @import("glob.zig");
const manifest = @import("manifest.zig");
const Allocator = std.mem.Allocator;
const Manifest = manifest.Manifest;
const Gate = manifest.Gate;

/// A gate never measured sorts after the cheap ones.
pub const unknown_seconds = 60.0;

/// A changed input and the families that reach it.
pub const Hit = struct {
    file: []const u8,
    families: u64,
};

pub const Selection = struct {
    /// Manifest order.
    gates: []const *const Gate,
    /// The families selected, each with the file that first selected it.
    reasons: []const Reason,
    /// The first file that selected the canaries, with its family count.
    shared: ?Hit,

    pub const Reason = struct {
        family: []const u8,
        file: []const u8,
    };
};

pub fn isShared(m: *const Manifest, families: u64) bool {
    const count = @popCount(families);
    return count == 0 or count > m.family_cutoff;
}

/// The selection `hits` make in `tier`.
pub fn impact(arena: Allocator, m: *const Manifest, hits: []const Hit, tier: []const u8) Allocator.Error!Selection {
    var families: u64 = 0;
    var reasons: std.ArrayList(Selection.Reason) = .empty;
    var shared: ?Hit = null;
    for (hits) |hit| {
        if (isShared(m, hit.families)) {
            if (shared == null) shared = hit;
            continue;
        }
        for (m.families, 0..) |family, i| {
            const bit = @as(u64, 1) << @intCast(i);
            if (hit.families & bit == 0 or families & bit != 0) continue;
            families |= bit;
            try reasons.append(arena, .{ .family = family.name, .file = hit.file });
        }
    }
    const verify = std.mem.eql(u8, tier, "verify");
    var gates: std.ArrayList(*const Gate) = .empty;
    for (m.gates) |*gate| {
        if (!std.mem.eql(u8, gate.tier, tier)) continue;
        const bit = @as(u64, 1) << (m.familyIndex(gate.family) orelse continue);
        const by_family = families & bit != 0;
        const by_share = shared != null and (!verify or isCanary(m, gate.name));
        if (by_family or by_share) try gates.append(arena, gate);
    }
    return .{ .gates = gates.items, .reasons = reasons.items, .shared = shared };
}

pub fn isCanary(m: *const Manifest, name: []const u8) bool {
    for (m.canaries) |canary| {
        if (std.mem.eql(u8, canary, name)) return true;
    }
    return false;
}

/// The `items` (records with `paths`) any changed file matches.
pub fn matching(arena: Allocator, comptime T: type, items: []const T, changed: []const []const u8) Allocator.Error![]const *const T {
    var out: std.ArrayList(*const T) = .empty;
    for (items) |*item| {
        for (changed) |path| {
            if (glob.matchesAny(item.paths, path)) {
                try out.append(arena, item);
                break;
            }
        }
    }
    return out.items;
}

pub const Requirement = struct {
    tier: []const u8,
    file: []const u8,
    reason: []const u8,
};

/// Each `requires` rule the change matches, with the first file that did.
pub fn requirements(arena: Allocator, rules: []const manifest.Rule, changed: []const []const u8) Allocator.Error![]const Requirement {
    var out: std.ArrayList(Requirement) = .empty;
    for (rules) |rule| {
        for (changed) |path| {
            if (glob.matchesAny(rule.paths, path)) {
                try out.append(arena, .{ .tier = rule.tier, .file = path, .reason = rule.reason });
                break;
            }
        }
    }
    return out.items;
}

/// A check's argv with `{zig-files}` expanded to the changed Zig sources;
/// null when it has none to read.
pub fn checkArgv(arena: Allocator, check: *const manifest.Check, changed: []const []const u8) Allocator.Error!?[]const []const u8 {
    var zig: std.ArrayList([]const u8) = .empty;
    for (changed) |path| {
        if (std.mem.endsWith(u8, path, ".zig") or std.mem.endsWith(u8, path, ".zon")) try zig.append(arena, path);
    }
    var out: std.ArrayList([]const u8) = .empty;
    for (check.command) |arg| {
        if (!std.mem.eql(u8, arg, "{zig-files}")) {
            try out.append(arena, arg);
            continue;
        }
        if (zig.items.len == 0) return null;
        try out.appendSlice(arena, zig.items);
    }
    return out.items;
}

/// Measured seconds by gate name.
pub const Times = std.StringHashMapUnmanaged(f64);

fn cost(times: *const Times, gate: *const Gate) f64 {
    return times.get(gate.name) orelse unknown_seconds;
}

/// Cheapest first, one model at a time: the files together outgrow memory, so
/// interleaving models reloads each from disk. Models go in the order of their
/// cheapest gate, a model's gates by their last measured seconds; ties keep
/// manifest order.
pub fn byCost(arena: Allocator, gates: []const *const Gate, times: *const Times) Allocator.Error![]const *const Gate {
    var cheapest: std.StringHashMapUnmanaged(f64) = .empty;
    for (gates) |gate| {
        const entry = try cheapest.getOrPut(arena, gate.model);
        const seconds = cost(times, gate);
        entry.value_ptr.* = if (entry.found_existing) @min(entry.value_ptr.*, seconds) else seconds;
    }
    const Order = struct {
        cheapest: *const std.StringHashMapUnmanaged(f64),
        times: *const Times,

        fn lessThan(order: @This(), a: *const Gate, b: *const Gate) bool {
            const first_a = order.cheapest.get(a.model).?;
            const first_b = order.cheapest.get(b.model).?;
            if (first_a != first_b) return first_a < first_b;
            switch (std.mem.order(u8, a.model, b.model)) {
                .lt => return true,
                .gt => return false,
                .eq => {},
            }
            return cost(order.times, a) < cost(order.times, b);
        }
    };
    const sorted = try arena.dupe(*const Gate, gates);
    std.mem.sort(*const Gate, sorted, Order{ .cheapest = &cheapest, .times = times }, Order.lessThan);
    return sorted;
}

/// Which binaries the gates run: the two nuclis builds, and the check tools
/// behind the zig placeholders, built up front so the build is timed apart
/// from the checks.
pub const Builds = struct {
    metal: bool = false,
    cpu: bool = false,
    checks_metal: bool = false,
    checks_cpu: bool = false,
};

pub fn buildsNeeded(gates: []const *const Gate) Builds {
    var needed: Builds = .{};
    for (gates) |gate| {
        for (gate.command) |arg| {
            if (std.mem.indexOf(u8, arg, "{nuclis}") != null) needed.metal = true;
            if (std.mem.indexOf(u8, arg, "{nuclis-cpu}") != null) needed.cpu = true;
            if (std.mem.indexOf(u8, arg, "{zig-metal}") != null) needed.checks_metal = true;
            if (std.mem.indexOf(u8, arg, "{zig-cpu}") != null) needed.checks_cpu = true;
        }
    }
    return needed;
}

const testing = std.testing;

fn sample(arena: Allocator) !Manifest {
    return manifest.load(arena, try manifest.parse(arena, manifest.sample));
}

fn names(arena: Allocator, gates: []const *const Gate) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (gates) |gate| {
        try out.appendSlice(arena, gate.name);
        try out.append(arena, ' ');
    }
    return out.items;
}

test "one family selects its gates of the tier" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    const m = try sample(a);
    const hits = [_]Hit{.{ .file = "x.zig", .families = 0b10 }};
    const selection = try impact(a, &m, &hits, "verify");
    try testing.expectEqualStrings("b-metal ", try names(a, selection.gates));
    try testing.expectEqualStrings("b", selection.reasons[0].family);
    try testing.expectEqualStrings("x.zig", selection.reasons[0].file);
    try testing.expect(selection.shared == null);
}

test "an empty or wide reach selects the canaries" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    const m = try sample(a);
    for ([_]u64{ 0, 0b11 }) |families| {
        const hits = [_]Hit{.{ .file = "s.zig", .families = families }};
        const selection = try impact(a, &m, &hits, "verify");
        try testing.expectEqualStrings("a-trace ", try names(a, selection.gates));
        try testing.expectEqualStrings("s.zig", selection.shared.?.file);
    }
}

test "another tier has no canaries: a shared change selects all of it" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    const m = try sample(a);
    const hits = [_]Hit{.{ .file = "s.zig", .families = 0 }};
    try testing.expectEqualStrings("a-draft-cpu ", try names(a, (try impact(a, &m, &hits, "verify-cpu")).gates));
}

test "no hits select nothing" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    const m = try sample(a);
    const selection = try impact(a, &m, &.{}, "verify");
    try testing.expectEqual(@as(usize, 0), selection.gates.len);
}

test "checks and requirements follow paths" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    const m = try sample(a);
    const changed = [_][]const u8{ "inference/src/backends/cpu/vector.zig", "docs/x.md" };
    const checks = try matching(a, manifest.Check, m.checks, &changed);
    try testing.expectEqual(@as(usize, 2), checks.len);
    const argv = (try checkArgv(a, &m.checks[0], &changed)).?;
    try testing.expectEqual(@as(usize, 4), argv.len);
    try testing.expectEqualStrings("inference/src/backends/cpu/vector.zig", argv[3]);
    try testing.expect(try checkArgv(a, &m.checks[0], &.{"docs/x.md"}) == null);
    const required = try requirements(a, m.requires, &changed);
    try testing.expectEqual(@as(usize, 1), required.len);
    try testing.expectEqualStrings("verify-cpu", required[0].tier);
    try testing.expectEqualStrings("inference/src/backends/cpu/vector.zig", required[0].file);
    try testing.expectEqual(@as(usize, 0), (try requirements(a, m.requires, &.{"src/tui/editor.zig"})).len);
}

fn stub(name: []const u8, model: []const u8) Gate {
    return .{
        .name = name,
        .family = "a",
        .tier = "verify",
        .model = model,
        .mtp = null,
        .baseline = null,
        .data = &.{},
        .command = &.{},
        .comparator = .exit,
        .evidence = "",
        .raw = .null,
    };
}

test "unmeasured gates run after the cheap ones" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    const gates = [_]Gate{ stub("slow", "a"), stub("new", "a"), stub("cheap", "a"), stub("mid", "a") };
    var times: Times = .empty;
    try times.put(a, "slow", 120);
    try times.put(a, "cheap", 0.5);
    try times.put(a, "mid", 10);
    const order = try byCost(a, &.{ &gates[0], &gates[1], &gates[2], &gates[3] }, &times);
    try testing.expectEqualStrings("cheap mid new slow ", try names(a, order));
}

test "a model's gates stay together" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    const gates = [_]Gate{ stub("a-big", "a"), stub("b-small", "b"), stub("a-small", "a"), stub("b-big", "b") };
    var times: Times = .empty;
    try times.put(a, "a-big", 50);
    try times.put(a, "a-small", 2);
    try times.put(a, "b-small", 0.5);
    try times.put(a, "b-big", 9);
    const order = try byCost(a, &.{ &gates[0], &gates[1], &gates[2], &gates[3] }, &times);
    try testing.expectEqualStrings("b-small b-big a-small a-big ", try names(a, order));
}

test "builds follow the placeholders" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    const m = try sample(a);
    const all = [_]*const Gate{ &m.gates[0], &m.gates[1] };
    try testing.expectEqual(Builds{ .metal = true, .checks_cpu = true }, buildsNeeded(&all));
    try testing.expectEqual(Builds{ .checks_cpu = true }, buildsNeeded(all[1..]));
}
