//! What each tier costs, from the last passing seconds `times.json` keeps:
//! `--costs` prints it as Markdown (docs/development.md § Gates pastes it
//! with its revision), and a tier run warns, or with `--strict` fails, when
//! its total passes the manifest's `budgets`. A gate never timed counts as
//! zero and is reported as untimed, so a total is a floor, never a guess.
const std = @import("std");
const manifest = @import("manifest.zig");
const select = @import("select.zig");
const Allocator = std.mem.Allocator;
const Gate = manifest.Gate;
const Manifest = manifest.Manifest;

pub const slowest_shown = 5;

pub const Cost = struct {
    name: []const u8,
    gates: usize,
    timed: usize,
    seconds: f64,
    budget: ?f64,
    /// Slowest first, at most `slowest_shown`.
    slowest: []const struct { []const u8, f64 },

    pub fn over(c: Cost) bool {
        return if (c.budget) |b| c.seconds > b else false;
    }
};

/// The cost of `gates` under the budget called `name`.
pub fn of(arena: Allocator, m: *const Manifest, name: []const u8, gates: []const *const Gate, times: *const select.Times) Allocator.Error!Cost {
    var timed: std.ArrayList(struct { []const u8, f64 }) = .empty;
    var total: f64 = 0;
    for (gates) |gate| {
        const seconds = times.get(gate.name) orelse continue;
        total += seconds;
        try timed.append(arena, .{ gate.name, seconds });
    }
    std.mem.sort(struct { []const u8, f64 }, timed.items, {}, struct {
        fn slower(_: void, a: struct { []const u8, f64 }, b: struct { []const u8, f64 }) bool {
            return a[1] > b[1] or (a[1] == b[1] and std.mem.lessThan(u8, a[0], b[0]));
        }
    }.slower);
    return .{ .name = name, .gates = gates.len, .timed = timed.items.len, .seconds = total, .budget = m.budget(name), .slowest = timed.items[0..@min(slowest_shown, timed.items.len)] };
}

/// The gates of `tier`, in manifest order.
pub fn tierGates(arena: Allocator, m: *const Manifest, tier: []const u8) Allocator.Error![]const *const Gate {
    var out: std.ArrayList(*const Gate) = .empty;
    for (m.gates) |*gate| {
        if (std.mem.eql(u8, gate.tier, tier)) try out.append(arena, gate);
    }
    return out.items;
}

/// The canaries, in their listed order.
pub fn canaryGates(arena: Allocator, m: *const Manifest) Allocator.Error![]const *const Gate {
    var out: std.ArrayList(*const Gate) = .empty;
    for (m.canaries) |name| {
        if (m.gate(name)) |gate| try out.append(arena, gate);
    }
    return out.items;
}

/// One row per tier and one for the canaries.
pub fn all(arena: Allocator, m: *const Manifest, times: *const select.Times) Allocator.Error![]const Cost {
    var out: std.ArrayList(Cost) = .empty;
    for (manifest.tiers) |tier| try out.append(arena, try of(arena, m, tier, try tierGates(arena, m, tier), times));
    try out.append(arena, try of(arena, m, "canaries", try canaryGates(arena, m), times));
    return out.items;
}

pub fn writeMarkdown(out: *std.Io.Writer, rows: []const Cost) std.Io.Writer.Error!void {
    try out.writeAll("| tier | gates | timed | seconds | budget | slowest |\n| --- | ---: | ---: | ---: | ---: | --- |\n");
    for (rows) |c| {
        try out.print("| `{s}` | {d} | {d} | {d:.0} | ", .{ c.name, c.gates, c.timed, c.seconds });
        if (c.budget) |b| try out.print("{d:.0}{s}", .{ b, if (c.over()) " (over)" else "" }) else try out.writeAll("—");
        try out.writeAll(" | ");
        for (c.slowest, 0..) |entry, i| try out.print("{s}`{s}` {d:.0} s", .{ if (i == 0) "" else ", ", entry[0], entry[1] });
        if (c.slowest.len == 0) try out.writeAll("—");
        try out.writeAll(" |\n");
    }
}

const testing = std.testing;

test "a tier's cost sums what was timed, slowest first, against its budget" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    var m = try manifest.load(a, try manifest.parse(a, manifest.sample));
    const budgets = [_]manifest.Budget{ .{ .name = "verify", .seconds = 10 }, .{ .name = "canaries", .seconds = 60 } };
    m.budgets = &budgets;
    var times: select.Times = .empty;
    try times.put(a, "a-trace", 8);
    try times.put(a, "b-metal", 4);
    const rows = try all(a, &m, &times);
    const verify = rows[0];
    try testing.expectEqualStrings("verify", verify.name);
    try testing.expectEqual(@as(usize, 2), verify.gates);
    try testing.expectEqual(@as(f64, 12), verify.seconds);
    try testing.expect(verify.over());
    try testing.expectEqualStrings("a-trace", verify.slowest[0][0]);
    const canaries = rows[rows.len - 1];
    try testing.expect(!canaries.over() and canaries.seconds == 8);
    // verify-cpu: one gate, never timed.
    try testing.expectEqual(@as(usize, 0), rows[3].timed);
    var buffer: std.Io.Writer.Allocating = .init(a);
    try writeMarkdown(&buffer.writer, rows);
    try testing.expect(std.mem.indexOf(u8, buffer.written(), "| `verify` | 2 | 2 | 12 | 10 (over) | `a-trace` 8 s, `b-metal` 4 s |") != null);
    try testing.expect(std.mem.indexOf(u8, buffer.written(), "| `verify-cpu` | 1 | 0 | 0 | — | — |") != null);
}
