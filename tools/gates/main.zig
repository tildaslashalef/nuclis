//! `nuclis-gates`: the gate runner. Parses the command line and dispatches;
//! `run.zig` holds the lanes. It builds through `zig build` itself, so it must
//! not run inside a `zig build` Run step. docs/development.md § Gates.
const std = @import("std");
const compare = @import("compare.zig");
const glob = @import("glob.zig");
const manifest = @import("manifest.zig");
const run = @import("run.zig");

const usage =
    \\usage: nuclis-gates MODE [options]
    \\       nuclis-gates compare NATIVE REFERENCE --positions N [options]
    \\
    \\modes:
    \\  --list              every gate with its tier, family, model, and evidence
    \\  --validate          validate gates.json against the tree (no model needed)
    \\  --auto [REV]        what a change needs: the checks its paths select, the
    \\                      gates its impact selects, cheapest first until one
    \\                      fails, then the tiers it requires. REV defaults to
    \\                      where the branch left origin/main
    \\  --tier TIER         every gate of a tier; with --changed, the selected ones
    \\  --gate NAME         gates by name or pattern (repeatable): 'qwen38-*'
    \\  --changed [REV]     the verify gates the working tree's change from REV
    \\                      selects (default HEAD); --tier for another tier
    \\  --sweep             every tier's gates not already cached, cheapest first
    \\  --replay A..B       what a committed range selects; runs nothing
    \\
    \\options:
    \\  --dry-run           print the commands instead of running them
    \\  --no-build          do not rebuild the binaries first
    \\  --no-cache          run gates even with a remembered pass
    \\  --json              print the results as JSON
    \\  --timeline          print each output line of a gate with its time
    \\  --keep-going        with --auto and --sweep, run on after a failure
    \\  --strict            a gate whose model or data is missing fails
    \\
    \\compare options:
    \\  --positions N --max-absolute X --max-relative-rms X --embedding N
    \\  --layers N --vocab N --draft
    \\
;

const Mode = union(enum) {
    help,
    list,
    validate,
    auto: ?[]const u8,
    tier: []const u8,
    gates: []const []const u8,
    changed: []const u8,
    sweep,
    replay: struct { from: []const u8, to: []const u8 },
};

const Command = struct {
    mode: Mode,
    tier: ?[]const u8 = null,
    options: run.Options = .{},
};

const UsageError = error{Usage} || std.mem.Allocator.Error;

fn isTier(name: []const u8) bool {
    for (manifest.tiers) |tier| {
        if (std.mem.eql(u8, tier, name)) return true;
    }
    return false;
}

/// Reports why a command line is refused; the caller exits 2.
const Diagnostic = struct {
    message: []const u8 = "",
};

fn fail(diag: *Diagnostic, message: []const u8) error{Usage} {
    diag.message = message;
    return error.Usage;
}

fn parse(arena: std.mem.Allocator, args: []const []const u8, diag: *Diagnostic) UsageError!Command {
    var command: Command = .{ .mode = .help };
    var mode: ?Mode = null;
    var gates: std.ArrayList([]const u8) = .empty;
    var changed: ?[]const u8 = null;
    var i: usize = 0;
    const optional = struct {
        fn next(list: []const []const u8, index: *usize) ?[]const u8 {
            if (index.* + 1 >= list.len or std.mem.startsWith(u8, list[index.* + 1], "-")) return null;
            index.* += 1;
            return list[index.*];
        }
    }.next;
    const required = struct {
        fn next(list: []const []const u8, index: *usize, d: *Diagnostic, flag: []const u8) error{Usage}![]const u8 {
            if (index.* + 1 >= list.len) {
                d.message = flag;
                return error.Usage;
            }
            index.* += 1;
            return list[index.*];
        }
    }.next;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            mode = .help;
        } else if (std.mem.eql(u8, arg, "--list")) {
            mode = .list;
        } else if (std.mem.eql(u8, arg, "--validate")) {
            mode = .validate;
        } else if (std.mem.eql(u8, arg, "--sweep")) {
            mode = .sweep;
        } else if (std.mem.eql(u8, arg, "--auto")) {
            mode = .{ .auto = optional(args, &i) };
        } else if (std.mem.eql(u8, arg, "--changed")) {
            changed = optional(args, &i) orelse "HEAD";
        } else if (std.mem.eql(u8, arg, "--replay")) {
            const range = try required(args, &i, diag, "--replay needs A..B");
            const dots = std.mem.indexOf(u8, range, "..") orelse return fail(diag, "--replay needs A..B");
            mode = .{ .replay = .{ .from = range[0..dots], .to = range[dots + 2 ..] } };
        } else if (std.mem.eql(u8, arg, "--gate")) {
            try gates.append(arena, try required(args, &i, diag, "--gate needs a name"));
        } else if (std.mem.eql(u8, arg, "--tier")) {
            const tier = try required(args, &i, diag, "--tier needs a tier");
            if (!isTier(tier)) return fail(diag, "--tier: unknown tier");
            command.tier = tier;
        } else if (std.mem.eql(u8, arg, "--dry-run")) {
            command.options.dry_run = true;
        } else if (std.mem.eql(u8, arg, "--no-build")) {
            command.options.no_build = true;
        } else if (std.mem.eql(u8, arg, "--no-cache")) {
            command.options.no_cache = true;
        } else if (std.mem.eql(u8, arg, "--json")) {
            command.options.json = true;
        } else if (std.mem.eql(u8, arg, "--timeline")) {
            command.options.timeline = true;
        } else if (std.mem.eql(u8, arg, "--keep-going")) {
            command.options.keep_going = true;
        } else if (std.mem.eql(u8, arg, "--strict")) {
            command.options.strict = true;
        } else {
            diag.message = try arena.print("unknown argument {s}", .{arg});
            return error.Usage;
        }
    }
    // `--auto` wins, then names, then a change, then a whole tier.
    if (mode) |m| {
        command.mode = m;
    } else if (gates.items.len > 0) {
        command.mode = .{ .gates = gates.items };
    } else if (changed) |rev| {
        command.mode = .{ .changed = rev };
    } else if (command.tier) |tier| {
        command.mode = .{ .tier = tier };
    } else {
        return fail(diag, "one of --list, --validate, --auto, --tier, --gate, --changed, --sweep is required");
    }
    return command;
}

fn parseCompare(args: []const []const u8, diag: *Diagnostic) error{Usage}!struct { []const u8, []const u8, compare.Options } {
    var options: compare.Options = .{ .positions = 0 };
    var directories: [2][]const u8 = undefined;
    var count: usize = 0;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--draft")) {
            options.draft = true;
            continue;
        }
        if (!std.mem.startsWith(u8, arg, "--")) {
            if (count == 2) return fail(diag, "compare takes NATIVE and REFERENCE");
            directories[count] = arg;
            count += 1;
            continue;
        }
        i += 1;
        if (i == args.len) return fail(diag, "compare: a flag needs a value");
        const value = args[i];
        const bad = "compare: a flag's value does not parse";
        if (std.mem.eql(u8, arg, "--positions")) {
            options.positions = std.fmt.parseInt(u32, value, 10) catch return fail(diag, bad);
        } else if (std.mem.eql(u8, arg, "--max-absolute")) {
            options.max_absolute = std.fmt.parseFloat(f64, value) catch return fail(diag, bad);
        } else if (std.mem.eql(u8, arg, "--max-relative-rms")) {
            options.max_relative_rms = std.fmt.parseFloat(f64, value) catch return fail(diag, bad);
        } else if (std.mem.eql(u8, arg, "--embedding")) {
            options.embedding = std.fmt.parseInt(u32, value, 10) catch return fail(diag, bad);
        } else if (std.mem.eql(u8, arg, "--layers")) {
            options.layers = std.fmt.parseInt(u32, value, 10) catch return fail(diag, bad);
        } else if (std.mem.eql(u8, arg, "--vocab")) {
            options.vocab = std.fmt.parseInt(u32, value, 10) catch return fail(diag, bad);
        } else {
            return fail(diag, "compare: unknown flag");
        }
    }
    if (count != 2) return fail(diag, "compare takes NATIVE and REFERENCE");
    if (options.positions < 1 or options.positions > compare.max_positions) {
        return fail(diag, "positions must be between 1 and 32768");
    }
    return .{ directories[0], directories[1], options };
}

fn runCompare(init: std.process.Init, args: []const []const u8, out: *std.Io.Writer, err: *std.Io.Writer) !u8 {
    var diag: Diagnostic = .{};
    const native, const reference, const options = parseCompare(args, &diag) catch {
        try err.print("error: {s}\n{s}", .{ diag.message, usage });
        return 2;
    };
    var where: compare.Diagnostic = .{};
    const report = compare.compare(init.arena.allocator(), init.gpa, init.io, native, reference, options, &where) catch |e| {
        try err.print("error: {s}: {s}\n", .{ @errorName(e), where.path });
        return 1;
    };
    try report.write(out);
    return if (report.passed) 0 else 1;
}

/// The nearest directory upward holding `gates.json`; the process moves there.
fn enterRoot(arena: std.mem.Allocator, io: std.Io) !void {
    var path: []const u8 = try std.process.currentPathAlloc(io, arena);
    while (true) {
        const candidate = try std.fs.path.join(arena, &.{ path, "gates.json" });
        if (std.Io.Dir.cwd().access(io, candidate, .{})) |_| {
            return std.process.setCurrentPath(io, path);
        } else |_| {}
        path = std.fs.path.dirname(path) orelse return error.NoManifest;
    }
}

fn loadManifest(arena: std.mem.Allocator, io: std.Io, err: *std.Io.Writer) !manifest.Manifest {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, "gates.json", arena, .limited(manifest.max_manifest_bytes));
    const doc = try manifest.parse(arena, text);
    const problems = try manifest.validate(arena, doc);
    if (problems.len > 0) {
        for (problems) |problem| try err.print("gates.json: {s}\n", .{problem});
        return error.InvalidManifest;
    }
    return manifest.load(arena, doc);
}

fn gatesNamed(arena: std.mem.Allocator, m: *const manifest.Manifest, patterns: []const []const u8, err: *std.Io.Writer) !?[]const *const manifest.Gate {
    var selected: std.ArrayList(*const manifest.Gate) = .empty;
    for (patterns) |pattern| {
        var hit = false;
        for (m.gates) |*gate| {
            if (!glob.matchesName(pattern, gate.name)) continue;
            hit = true;
            if (std.mem.indexOfScalar(*const manifest.Gate, selected.items, gate) == null) try selected.append(arena, gate);
        }
        if (!hit) {
            try err.print("no gate matches '{s}'; --list shows them\n", .{pattern});
            return null;
        }
    }
    return selected.items;
}

fn dispatch(init: std.process.Init, args: []const []const u8, out: *std.Io.Writer, err: *std.Io.Writer) !u8 {
    const arena = init.arena.allocator();
    const io = init.io;
    var diag: Diagnostic = .{};
    const command = parse(arena, args, &diag) catch |e| switch (e) {
        error.Usage => {
            try err.print("error: {s}\n{s}", .{ diag.message, usage });
            return 2;
        },
        else => |other| return other,
    };
    if (command.mode == .help) {
        try out.writeAll(usage);
        return 0;
    }
    try enterRoot(arena, io);
    const m = loadManifest(arena, io, err) catch |e| switch (e) {
        error.InvalidManifest => return 2,
        else => |other| return other,
    };
    if (command.mode == .list) {
        for (m.gates) |gate| {
            try out.print("{s:<34} {s:<14} {s:<14} {s:<20} {s}\n", .{ gate.name, gate.tier, gate.family, gate.model, gate.evidence });
        }
        return 0;
    }
    // A check's `make` must not inherit this run's `ARGS=...`.
    for ([_][]const u8{ "MAKEFLAGS", "MFLAGS", "MAKEOVERRIDES", "ARGS", "BASE", "NAME" }) |name| _ = init.environ_map.swapRemove(name);
    var session: run.Session = .{
        .arena = arena,
        .gpa = init.gpa,
        .io = io,
        .env = init.environ_map,
        .m = &m,
        .options = command.options,
        .out = out,
        .err = err,
    };
    try session.open();
    switch (command.mode) {
        .help, .list => unreachable,
        .validate => {
            const status = try session.lint();
            if (status == 0) try out.print("gates.json: {d} gates, {d} models, {d} families, valid\n", .{ m.gates.len, m.models.len, m.families.len });
            return status;
        },
        .auto => |rev| return session.auto(rev),
        .sweep => return session.sweep(),
        .changed => |rev| return session.changed(rev, command.tier orelse "verify"),
        .replay => |range| return session.replay(range.from, range.to),
        .tier => |tier| {
            var selected: std.ArrayList(*const manifest.Gate) = .empty;
            for (m.gates) |*gate| {
                if (std.mem.eql(u8, gate.tier, tier)) try selected.append(arena, gate);
            }
            return session.runAll(selected.items);
        },
        .gates => |patterns| {
            const selected = try gatesNamed(arena, &m, patterns, err) orelse return 2;
            return session.runAll(selected);
        },
    }
}

pub fn main(init: std.process.Init) void {
    var out_buffer: [4096]u8 = undefined;
    var err_buffer: [1024]u8 = undefined;
    var out_writer = std.Io.File.stdout().writerStreaming(init.io, &out_buffer);
    var err_writer = std.Io.File.stderr().writerStreaming(init.io, &err_buffer);
    const out = &out_writer.interface;
    const err = &err_writer.interface;
    const status = status: {
        const args = init.minimal.args.toSlice(init.arena.allocator()) catch break :status 1;
        const rest = args[1..];
        const result = if (rest.len > 0 and std.mem.eql(u8, rest[0], "compare"))
            runCompare(init, rest[1..], out, err)
        else
            dispatch(init, rest, out, err);
        break :status result catch |e| {
            err.print("error: {s}\n", .{@errorName(e)}) catch {};
            break :status 1;
        };
    };
    out.flush() catch {};
    err.flush() catch {};
    std.process.exit(status);
}

test {
    _ = compare;
    _ = glob;
    _ = manifest;
    _ = run;
    _ = @import("cache.zig");
    _ = @import("fingerprint.zig");
    _ = @import("git.zig");
    _ = @import("select.zig");
    _ = @import("tree.zig");
}

const testing = std.testing;

fn parsed(arena: std.mem.Allocator, args: []const []const u8) !Command {
    var diag: Diagnostic = .{};
    return parse(arena, args, &diag);
}

test "a revision after --auto and --changed is optional" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    try testing.expect((try parsed(a, &.{ "--auto", "--dry-run" })).mode.auto == null);
    try testing.expectEqualStrings("abc", (try parsed(a, &.{ "--auto", "abc" })).mode.auto.?);
    try testing.expect((try parsed(a, &.{ "--auto", "--dry-run" })).options.dry_run);
    try testing.expectEqualStrings("HEAD", (try parsed(a, &.{"--changed"})).mode.changed);
    try testing.expectEqualStrings("main", (try parsed(a, &.{ "--changed", "main", "--tier", "verify-cpu" })).mode.changed);
}

test "names, a change, and a tier take precedence in that order" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    const command = try parsed(a, &.{ "--tier", "verify", "--gate", "a", "--gate", "b*" });
    try testing.expectEqual(@as(usize, 2), command.mode.gates.len);
    try testing.expectEqualStrings("verify-long", (try parsed(a, &.{ "--tier", "verify-long" })).mode.tier);
    try testing.expectError(error.Usage, parsed(a, &.{ "--tier", "cheap" }));
    try testing.expectError(error.Usage, parsed(a, &.{"--dry-run"}));
    try testing.expectError(error.Usage, parsed(a, &.{"--bogus"}));
    const range = (try parsed(a, &.{ "--replay", "aa..bb" })).mode.replay;
    try testing.expectEqualStrings("aa", range.from);
    try testing.expectEqualStrings("bb", range.to);
}

test "compare parses as its documented flags" {
    var diag: Diagnostic = .{};
    const native, const reference, const options = try parseCompare(&.{ "n", "r", "--positions", "2", "--max-absolute", "0.5", "--draft", "--vocab", "7" }, &diag);
    try testing.expectEqualStrings("n", native);
    try testing.expectEqualStrings("r", reference);
    try testing.expectEqual(@as(u32, 2), options.positions);
    try testing.expectEqual(@as(f64, 0.5), options.max_absolute);
    try testing.expectEqual(@as(u32, 7), options.vocab);
    try testing.expect(options.draft);
    try testing.expectError(error.Usage, parseCompare(&.{ "n", "r" }, &diag));
    try testing.expectError(error.Usage, parseCompare(&.{ "n", "--positions", "1" }, &diag));
    try testing.expectError(error.Usage, parseCompare(&.{ "n", "r", "--positions", "40000" }, &diag));
}
