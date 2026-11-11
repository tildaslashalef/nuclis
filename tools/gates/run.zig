//! The lanes: what each command selects, and running it. A gate is one command
//! (argv with placeholders) plus a comparator: `exit` (the command's status
//! decides) or `trace` (its `{trace}` directory goes through `compare.zig`
//! against the gate's bounds). Selected gates are triaged first (no model or
//! data: skipped; a remembered pass: cached), so a run with nothing left does
//! not build. docs/development.md § Gates.
const std = @import("std");
const compare = @import("compare.zig");
const cache = @import("cache.zig");
const fingerprint = @import("fingerprint.zig");
const git = @import("git.zig");
const glob = @import("glob.zig");
const manifest = @import("manifest.zig");
const select = @import("select.zig");
const tree_module = @import("tree.zig");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const Dir = Io.Dir;
const Environ = std.process.Environ;
const Manifest = manifest.Manifest;
const Gate = manifest.Gate;
const Tree = tree_module.Tree;

pub const trace_root = ".zig-cache/gates/trace";
const cpu_prefix = ".zig-cache/gates/cpu";
/// Of a gate's output, the newest lines kept for `--timeline` and `--json`.
const max_log_lines = 4096;
const max_line_bytes = 4096;
const tail_lines = 20;
/// A clip listing past this is not one.
const max_listing_bytes = 16 << 20;

pub const Options = struct {
    dry_run: bool = false,
    no_build: bool = false,
    json: bool = false,
    timeline: bool = false,
    keep_going: bool = false,
    strict: bool = false,
    no_cache: bool = false,
};

// ---- placeholders, models, data --------------------------------------------

/// The pinned path, or the `<KEY>_MODEL` environment override; `~` expanded.
pub fn modelPath(arena: Allocator, m: *const Manifest, key: []const u8, env: *const Environ.Map) Allocator.Error![]const u8 {
    const name = try arena.print("{s}_MODEL", .{key});
    const upper = try std.ascii.allocUpperString(arena, name);
    const path = env.get(upper) orelse m.model(key).?.path;
    return expandUser(arena, path, env);
}

fn expandUser(arena: Allocator, path: []const u8, env: *const Environ.Map) Allocator.Error![]const u8 {
    if (!std.mem.startsWith(u8, path, "~/")) return path;
    const home = env.get("HOME") orelse return path;
    return std.fs.path.join(arena, &.{ home, path[2..] });
}

fn usesPlaceholder(gate: *const Gate, token: []const u8) bool {
    for (gate.command) |arg| {
        if (std.mem.indexOf(u8, arg, token) != null) return true;
    }
    return false;
}

/// The model files a gate reads: `model`, `mtp`, `baseline`, and the entry's
/// `mmproj` when its command names it. A command that runs `{entry}` reads
/// the pinned files the catalogue resolves, so no override applies to it.
pub fn modelFiles(arena: Allocator, m: *const Manifest, gate: *const Gate, env: *const Environ.Map) Allocator.Error![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    const by_entry = usesPlaceholder(gate, "{entry}");
    const model = m.model(gate.model).?;
    try out.append(arena, if (by_entry) try expandUser(arena, model.path, env) else try modelPath(arena, m, gate.model, env));
    if (gate.mtp) |key| try out.append(arena, try modelPath(arena, m, key, env));
    if (gate.baseline) |key| try out.append(arena, try modelPath(arena, m, key, env));
    if (model.mmproj) |mmproj| {
        if (by_entry or usesPlaceholder(gate, "{mmproj}")) try out.append(arena, try expandUser(arena, mmproj, env));
    }
    return out.items;
}

fn zigBuild(arena: Allocator, out: *std.ArrayList([]const u8), m: *const Manifest, flavour: enum { metal, cpu }) Allocator.Error!void {
    try out.appendSlice(arena, &.{ "zig", "build" });
    switch (flavour) {
        .metal => {
            try out.append(arena, "-Dmetal=true");
            try out.append(arena, try arena.print("-Doptimize={s}", .{m.metal_optimize}));
        },
        .cpu => try out.append(arena, try arena.print("-Doptimize={s}", .{m.cpu_optimize})),
    }
}

/// Substitutes placeholders; the build placeholders expand to several items.
pub fn expand(arena: Allocator, m: *const Manifest, gate: *const Gate, env: *const Environ.Map) Allocator.Error![]const []const u8 {
    const Scalar = struct { []const u8, ?[]const u8 };
    const entry = m.model(gate.model).?;
    const scalars = [_]Scalar{
        .{ "{trace}", try arena.print("{s}/{s}", .{ trace_root, gate.name }) },
        .{ "{model}", try modelPath(arena, m, gate.model, env) },
        .{ "{mtp}", if (gate.mtp) |key| try modelPath(arena, m, key, env) else null },
        .{ "{baseline}", if (gate.baseline) |key| try modelPath(arena, m, key, env) else null },
        .{ "{mmproj}", if (entry.mmproj) |path| try expandUser(arena, path, env) else null },
        .{ "{entry}", entry.entry },
    };
    var out: std.ArrayList([]const u8) = .empty;
    for (gate.command) |arg| {
        if (std.mem.eql(u8, arg, "{nuclis}")) {
            try out.append(arena, "./zig-out/bin/nuclis");
        } else if (std.mem.eql(u8, arg, "{nuclis-cpu}")) {
            try out.append(arena, "./" ++ cpu_prefix ++ "/bin/nuclis");
        } else if (std.mem.eql(u8, arg, "{zig-metal}")) {
            try zigBuild(arena, &out, m, .metal);
        } else if (std.mem.eql(u8, arg, "{zig-cpu}")) {
            try zigBuild(arena, &out, m, .cpu);
        } else {
            var text = arg;
            for (scalars) |scalar| {
                if (scalar[1]) |value| text = try std.mem.replaceOwned(u8, arena, text, scalar[0], value);
            }
            try out.append(arena, text);
        }
    }
    return out.items;
}

fn exists(io: Io, path: []const u8) bool {
    Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

pub const MissingData = struct {
    name: []const u8,
    absent: usize,
    total: usize,
    fetch: []const u8,
};

/// The files of data set `set`, under `root`: its `paths`, or every clip
/// `file` its `listed_in` manifest names.
fn dataFiles(arena: Allocator, io: Io, set: *const manifest.DataSet, root: []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    if (set.paths.len > 0) {
        for (set.paths) |path| try out.append(arena, try std.fs.path.join(arena, &.{ root, path }));
        return out.items;
    }
    const listing = try std.fs.path.join(arena, &.{ root, set.listed_in.? });
    const text = try Dir.cwd().readFileAlloc(io, listing, arena, .limited(max_listing_bytes));
    const doc = try manifest.parse(arena, text);
    const clips = if (doc == .object) doc.object.get("clips") else null;
    if (clips == null or clips.? != .array) return error.InvalidManifest;
    for (clips.?.array.items) |clip| {
        const file = if (clip == .object) clip.object.get("file") else null;
        if (file == null or file.? != .string) return error.InvalidManifest;
        try out.append(arena, try std.fs.path.join(arena, &.{ root, set.root.?, file.?.string }));
    }
    return out.items;
}

/// For each data set the gate reads with files not on this machine: its name,
/// how many of its files are missing, and the command that fetches it.
pub fn missingData(arena: Allocator, io: Io, m: *const Manifest, gate: *const Gate, root: []const u8) ![]const MissingData {
    var out: std.ArrayList(MissingData) = .empty;
    for (gate.data) |name| {
        const set = m.dataSet(name).?;
        const files = try dataFiles(arena, io, set, root);
        var absent: usize = 0;
        for (files) |file| {
            if (!exists(io, file)) absent += 1;
        }
        if (absent > 0) try out.append(arena, .{ .name = name, .absent = absent, .total = files.len, .fetch = set.fetch });
    }
    return out.items;
}

// ---- the engine tree on disk ------------------------------------------------

/// A `tree.build` reader: each read frees the last, so fixtures do not pile up.
const Disk = struct {
    gpa: Allocator,
    io: Io,
    held: ?[]u8 = null,

    pub fn read(disk: *Disk, path: []const u8) ![]const u8 {
        disk.deinit();
        disk.held = try Dir.cwd().readFileAlloc(disk.io, path, disk.gpa, .limited(tree_module.max_input_bytes));
        return disk.held.?;
    }

    fn deinit(disk: *Disk) void {
        if (disk.held) |bytes| disk.gpa.free(bytes);
        disk.held = null;
    }
};

fn listFiles(arena: Allocator, io: Io, root: []const u8, out: *std.ArrayList([]const u8)) !void {
    var dir = Dir.cwd().openDir(io, root, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return,
        else => |e| return e,
    };
    defer dir.close(io);
    var walker = try dir.walkSelectively(arena);
    defer walker.deinit();
    while (try walker.next(io)) |entry| {
        if (entry.kind == .directory) {
            if (!std.mem.eql(u8, entry.basename, ".zig-cache") and !std.mem.eql(u8, entry.basename, "zig-out")) {
                try walker.enter(io, entry);
            }
            continue;
        }
        try out.append(arena, try std.fs.path.join(arena, &.{ root, entry.path }));
    }
}

/// Every file the tree may hold, sorted.
fn universeOnDisk(arena: Allocator, io: Io) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    for ([_][]const u8{ "inference", "tests/fixtures", "scripts/gate", "src", "huggingface" }) |root| {
        try listFiles(arena, io, root, &out);
    }
    for ([_][]const u8{ "build.zig", "build.zig.zon" }) |path| {
        if (exists(io, path)) try out.append(arena, path);
    }
    std.mem.sort([]const u8, out.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lessThan);
    return out.items;
}

/// The compiler the gates' builds run, or "unknown"; a pass outlives no upgrade.
fn zigVersion(arena: Allocator, io: Io, env: *const Environ.Map) Allocator.Error![]const u8 {
    const result = std.process.run(arena, io, .{
        .argv = &.{ "zig", "version" },
        .environ_map = env,
        .stdout_limit = .limited(1 << 10),
        .stderr_limit = .limited(1 << 10),
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return "unknown",
    };
    if (result.term != .exited or result.term.exited != 0) return "unknown";
    return std.mem.trim(u8, result.stdout, " \n");
}

// ---- results ----------------------------------------------------------------

pub const Measured = struct {
    max_absolute: f64,
    max_relative_rms: f64,
    files: usize,
    greedy: ?[]const i64 = null,
    reference_greedy: ?[]const i64 = null,
};

pub const Line = struct {
    /// Seconds since launch.
    t: f64,
    text: []const u8,
};

pub const Result = struct {
    name: []const u8,
    tier: []const u8,
    command: []const []const u8,
    /// Null for a skip: the run neither passes nor fails it.
    passed: ?bool = null,
    seconds: f64 = 0,
    skipped: ?[]const u8 = null,
    /// The revision of the remembered pass.
    cached: ?[]const u8 = null,
    exit_status: ?i32 = null,
    detail: ?[]const u8 = null,
    measured: ?Measured = null,
    timeline: ?[]const Line = null,
};

const Builds = struct {
    metal: ?f64 = null,
    cpu: ?f64 = null,
    checks_metal: ?f64 = null,
    checks_cpu: ?f64 = null,

    fn total(builds: Builds) f64 {
        return (builds.metal orelse 0) + (builds.cpu orelse 0) + (builds.checks_metal orelse 0) + (builds.checks_cpu orelse 0);
    }

    fn any(builds: Builds) bool {
        return builds.metal != null or builds.cpu != null or builds.checks_metal != null or builds.checks_cpu != null;
    }
};

const Tally = struct {
    failed: std.ArrayList([]const u8) = .empty,
    skipped: std.ArrayList([]const u8) = .empty,
    cached: usize = 0,
};

// ---- the session ------------------------------------------------------------

pub const Session = struct {
    arena: Allocator,
    gpa: Allocator,
    io: Io,
    env: *Environ.Map,
    m: *const Manifest,
    options: Options,
    out: *Io.Writer,
    err: *Io.Writer,
    tree: Tree = undefined,
    universe: []const []const u8 = &.{},
    passes: cache.Passes = undefined,
    times: cache.TimesFile = undefined,
    revision: []const u8 = "unknown",
    /// `zig version`, part of every key.
    toolchain: []const u8 = "unknown",
    builds: Builds = .{},
    results: std.ArrayList(Result) = .empty,

    /// Loads the tree and the store; every lane needs both.
    pub fn open(s: *Session) !void {
        s.universe = try universeOnDisk(s.arena, s.io);
        var disk: Disk = .{ .gpa = s.gpa, .io = s.io };
        defer disk.deinit();
        s.tree = try tree_module.build(s.arena, s.gpa, s.m, s.universe, &disk);
        const dir = try cache.directory(s.arena, s.env);
        s.passes = try cache.Passes.load(s.arena, s.io, dir);
        s.times = try cache.TimesFile.load(s.arena, s.io, dir);
        s.revision = try git.revision(s.arena, s.io);
        s.toolchain = try zigVersion(s.arena, s.io, s.env);
    }

    fn say(s: *Session, comptime format: []const u8, args: anytype) !void {
        const w = if (s.options.json) s.err else s.out;
        try w.print(format ++ "\n", args);
        try w.flush();
    }

    fn note(s: *Session, comptime format: []const u8, args: anytype) !void {
        try s.err.print(format ++ "\n", args);
        try s.err.flush();
    }

    fn join(s: *Session, items: []const []const u8, separator: []const u8) ![]const u8 {
        return std.mem.join(s.arena, separator, items);
    }

    fn gateNames(s: *Session, gates: []const *const Gate) ![]const u8 {
        const names = try s.arena.alloc([]const u8, gates.len);
        for (names, gates) |*slot, gate| slot.* = gate.name;
        return s.join(names, ", ");
    }

    // ---- what changed -----------------------------------------------------

    pub const Change = struct {
        /// Every file that differs.
        files: []const []const u8,
        /// `files` without inputs whose fingerprint did not move.
        effective: []const []const u8,
        hits: []const select.Hit,
    };

    /// Sorts `files` into a change: inputs by fingerprint at `from` against
    /// `to` (null: the working tree).
    fn classify(s: *Session, files: []const []const u8, from: []const u8, to: ?[]const u8) !Change {
        var effective: std.ArrayList([]const u8) = .empty;
        var hits: std.ArrayList(select.Hit) = .empty;
        for (files) |path| {
            if (!tree_module.isInput(&s.tree, s.m, path)) {
                try effective.append(s.arena, path);
                continue;
            }
            const before = try s.digestAt(from, path);
            const after = if (to) |rev|
                try s.digestAt(rev, path)
            else if (s.tree.get(path)) |file|
                file.digest
            else
                null;
            if (before != null and after != null and std.mem.eql(u8, &before.?, &after.?)) continue;
            if (before == null and after == null) continue;
            try effective.append(s.arena, path);
            try hits.append(s.arena, .{ .file = path, .families = s.tree.familiesOf(s.m, path) });
        }
        return .{ .files = files, .effective = effective.items, .hits = hits.items };
    }

    fn digestAt(s: *Session, rev: []const u8, path: []const u8) !?fingerprint.Digest {
        const bytes = try git.show(s.arena, s.io, rev, path) orelse return null;
        return try fingerprint.ofFile(s.gpa, path, bytes);
    }

    fn changeSince(s: *Session, rev: []const u8) !Change {
        const files = try git.changedFiles(s.arena, s.io, rev) orelse {
            try s.note("git refuses {s}", .{rev});
            return error.BadRevision;
        };
        return s.classify(files, rev, null);
    }

    fn sayImpact(s: *Session, selection: select.Selection, tier: []const u8) !void {
        for (selection.reasons) |reason| try s.say("family {s}: {s}", .{ reason.family, reason.file });
        const hit = selection.shared orelse return;
        const label = if (std.mem.eql(u8, tier, "verify")) "canaries" else "the whole tier";
        const count = @popCount(hit.families);
        if (count == 0) {
            try s.say("{s}: {s} belongs to no family", .{ label, hit.file });
        } else {
            try s.say("{s}: {s} reaches {d} families", .{ label, hit.file, count });
        }
    }

    // ---- triage and execution ---------------------------------------------

    const Entry = struct {
        gate: *const Gate,
        key: cache.Key = undefined,
        result: Result,
        state: enum { run, skipped, cached },
    };

    fn triage(s: *Session, gates: []const *const Gate) ![]Entry {
        const entries = try s.arena.alloc(Entry, gates.len);
        for (entries, gates) |*entry, gate| {
            entry.* = .{
                .gate = gate,
                .result = .{ .name = gate.name, .tier = gate.tier, .command = try expand(s.arena, s.m, gate, s.env) },
                .state = .run,
            };
            const files = try modelFiles(s.arena, s.m, gate, s.env);
            var missing: std.ArrayList([]const u8) = .empty;
            for (files) |path| {
                if (!exists(s.io, path)) try missing.append(s.arena, path);
            }
            if (missing.items.len > 0) {
                entry.result.skipped = try s.arena.print("no model: {s}", .{try s.join(missing.items, ", ")});
                entry.state = .skipped;
                continue;
            }
            const absent = try missingData(s.arena, s.io, s.m, gate, ".");
            if (absent.len > 0) {
                var parts: std.ArrayList([]const u8) = .empty;
                for (absent) |set| {
                    try parts.append(s.arena, try s.arena.print("{s} ({d} of {d} files missing; {s})", .{ set.name, set.absent, set.total, set.fetch }));
                }
                entry.result.skipped = try s.arena.print("no data: {s}", .{try s.join(parts.items, "; ")});
                entry.state = .skipped;
                continue;
            }
            entry.key = try cache.key(s.arena, s.m, &s.tree, gate, try cache.modelIdentity(s.arena, s.io, files), s.toolchain);
            if (s.options.no_cache) continue;
            if (s.passes.get(&entry.key)) |pass| {
                entry.result.passed = true;
                entry.result.cached = pass.revision;
                entry.result.seconds = pass.seconds;
                entry.state = .cached;
            }
        }
        return entries;
    }

    fn zigEnv(s: *Session) !void {
        if (s.env.get("ZIG_GLOBAL_CACHE_DIR") != null) return;
        // `zig build` takes the global cache from the environment only;
        // every build and gate inherits it.
        const root = try std.process.currentPathAlloc(s.io, s.arena);
        try s.env.put("ZIG_GLOBAL_CACHE_DIR", try std.fs.path.join(s.arena, &.{ root, s.m.cache }));
    }

    fn build(s: *Session, argv: []const []const u8) !f64 {
        try s.note("build: {s}", .{try s.join(argv, " ")});
        if (s.options.dry_run) return 0;
        const started = Io.Clock.awake.now(s.io);
        var child = try std.process.spawn(s.io, .{ .argv = argv, .environ_map = s.env });
        defer child.kill(s.io);
        const term = try child.wait(s.io);
        if (term != .exited or term.exited != 0) return error.BuildFailed;
        return secondsSince(s.io, started);
    }

    fn buildFor(s: *Session, gates: []const *const Gate) !void {
        const needed = select.buildsNeeded(gates);
        var argv: std.ArrayList([]const u8) = .empty;
        if (needed.metal) {
            argv = .empty;
            try zigBuild(s.arena, &argv, s.m, .metal);
            s.builds.metal = try s.build(argv.items);
        }
        if (needed.cpu) {
            // The Metal flag stays on: the CPU gates read the same binary's
            // CPU backend, built at the CPU tier's optimize mode.
            argv = .empty;
            try argv.appendSlice(s.arena, &.{ "zig", "build", "-Dmetal=true" });
            try argv.append(s.arena, try s.arena.print("-Doptimize={s}", .{s.m.cpu_optimize}));
            try argv.appendSlice(s.arena, &.{ "--prefix", cpu_prefix });
            s.builds.cpu = try s.build(argv.items);
        }
        // The placeholders' own flags, so the gates' `zig build` finds the compile cached.
        if (needed.checks_metal) {
            argv = .empty;
            try zigBuild(s.arena, &argv, s.m, .metal);
            try argv.appendSlice(s.arena, &.{ "check-tools", "--prefix", ".zig-cache/gates/checks-metal" });
            s.builds.checks_metal = try s.build(argv.items);
        }
        if (needed.checks_cpu) {
            argv = .empty;
            try zigBuild(s.arena, &argv, s.m, .cpu);
            try argv.appendSlice(s.arena, &.{ "check-tools", "--prefix", ".zig-cache/gates/checks-cpu" });
            s.builds.checks_cpu = try s.build(argv.items);
        }
    }

    fn sayDry(s: *Session, entry: *const Entry) !void {
        const gate = entry.gate;
        switch (entry.state) {
            .skipped => try s.say("{s}: skipped ({s})", .{ gate.name, entry.result.skipped.? }),
            .cached => try s.say("{s}: cached ({s}, {d:.1} s)", .{ gate.name, entry.result.cached.?, entry.result.seconds }),
            .run => {
                try s.say("{s}: {s}", .{ gate.name, try s.join(entry.result.command, " ") });
                if (gate.comparator == .trace) {
                    const t = gate.comparator.trace;
                    const shape = if (t.draft)
                        try s.arena.print("--draft --embedding {d}", .{t.embedding})
                    else
                        try s.arena.print("--embedding {d} --layers {d} --vocab {d}", .{ t.embedding, t.layers, t.vocab });
                    try s.say("  then nuclis-gates compare {s}/{s} {s} --positions {d} {s} --max-absolute {d} --max-relative-rms {d}", .{ trace_root, gate.name, t.reference, t.positions, shape, t.max_absolute, t.max_relative_rms });
                }
            },
        }
    }

    fn runGate(s: *Session, entry: *Entry) !void {
        const gate = entry.gate;
        const result = &entry.result;
        const trace = try s.arena.print("{s}/{s}", .{ trace_root, gate.name });
        try Dir.cwd().deleteTree(s.io, trace);
        try Dir.cwd().createDirPath(s.io, trace);
        const started = Io.Clock.awake.now(s.io);
        const captured = capture(s.arena, s.gpa, s.io, s.env, result.command) catch |err| {
            result.passed = false;
            result.detail = try s.arena.print("{s}: {s}", .{ result.command[0], @errorName(err) });
            return;
        };
        result.seconds = secondsSince(s.io, started);
        result.exit_status = captured.status;
        result.timeline = captured.lines;
        const tail = captured.lines[captured.lines.len -| tail_lines..];
        if (captured.status != 0) {
            result.passed = false;
            result.detail = try tailText(s.arena, tail);
            return;
        }
        switch (gate.comparator) {
            .exit => {
                result.passed = true;
                result.detail = if (tail.len > 0) tail[tail.len - 1].text else "";
            },
            .trace => |t| {
                var diag: compare.Diagnostic = .{};
                const report = compare.compare(s.arena, s.gpa, s.io, trace, t.reference, .{
                    .positions = t.positions,
                    .max_absolute = t.max_absolute,
                    .max_relative_rms = t.max_relative_rms,
                    .embedding = t.embedding,
                    .layers = t.layers,
                    .vocab = t.vocab,
                    .draft = t.draft,
                }, &diag) catch |err| {
                    result.seconds = secondsSince(s.io, started);
                    result.passed = false;
                    result.detail = try s.arena.print("{s}: {s}", .{ @errorName(err), diag.path });
                    return;
                };
                result.seconds = secondsSince(s.io, started);
                result.passed = report.passed;
                result.measured = .{
                    .max_absolute = report.maxAbsolute(),
                    .max_relative_rms = report.maxRelativeRms(),
                    .files = report.comparisons.len,
                    .greedy = report.native_greedy,
                    .reference_greedy = report.reference_greedy,
                };
            },
        }
    }

    fn sayResult(s: *Session, entry: *const Entry) !void {
        if (s.options.json) return;
        const r = &entry.result;
        if (r.skipped) |reason| return s.say("SKIP {s:<34} {s}", .{ r.name, reason });
        if (r.cached) |revision| return s.say("CACHED {s:<32} {d:>7.1} s  passed at {s}", .{ r.name, r.seconds, revision });
        const passed = r.passed.?;
        var line: Io.Writer.Allocating = .init(s.arena);
        const w = &line.writer;
        try w.print("{s} {s:<34} {d:>7.1} s", .{ if (passed) "PASS" else "FAIL", r.name, r.seconds });
        if (r.measured) |measured| {
            const t = entry.gate.comparator.trace;
            try w.print("  max abs {e:.3} <= {d}  rel rms {e:.3} <= {d}  ({d} files)", .{ measured.max_absolute, t.max_absolute, measured.max_relative_rms, t.max_relative_rms, measured.files });
            if (measured.greedy) |greedy| {
                try w.print("  greedy {any}", .{greedy});
                if (!std.mem.eql(i64, greedy, measured.reference_greedy.?)) try w.print(" != {any}", .{measured.reference_greedy.?});
            }
        } else if (passed and r.detail != null) {
            try w.print("  {s}", .{r.detail.?});
        }
        try s.say("{s}", .{line.written()});
        if (s.options.timeline) {
            for (r.timeline orelse &.{}) |timed| {
                try s.say("    {d:>7.1} s  {s}", .{ timed.t, timed.text[0..@min(timed.text.len, 160)] });
            }
        }
        if (!passed) {
            if (r.detail) |detail| {
                var it = std.mem.splitScalar(u8, detail, '\n');
                while (it.next()) |text| try s.say("    {s}", .{text});
            }
        }
    }

    /// Runs what triage left; `stopping` ends at the first failure unless
    /// `--keep-going`. Records every result in `tally` and the session.
    fn execute(s: *Session, gates: []const *const Gate, tally: *Tally, stopping: bool) !void {
        if (gates.len == 0) return;
        try s.zigEnv();
        const entries = try s.triage(gates);
        var to_run: std.ArrayList(*const Gate) = .empty;
        for (entries) |*entry| {
            if (entry.state == .run) try to_run.append(s.arena, entry.gate);
            if (entry.state == .cached) tally.cached += 1;
        }
        try s.say("{d} gate(s): {d} cached, {d} to run, {d} skipped", .{ gates.len, tally.cached, to_run.items.len, gates.len - tally.cached - to_run.items.len });
        if (!s.options.no_build) try s.buildFor(to_run.items);
        if (s.options.dry_run) {
            for (entries) |*entry| try s.sayDry(entry);
            return;
        }
        var seconds: std.ArrayList(struct { []const u8, f64 }) = .empty;
        var stopped = false;
        for (entries) |*entry| {
            if (stopped) break;
            switch (entry.state) {
                .cached => {},
                .skipped => {
                    try tally.skipped.append(s.arena, entry.gate.name);
                    if (s.options.strict) try tally.failed.append(s.arena, entry.gate.name);
                },
                .run => {
                    try s.runGate(entry);
                    try seconds.append(s.arena, .{ entry.gate.name, entry.result.seconds });
                    if (entry.result.passed.?) {
                        try s.passes.put(s.arena, &entry.key, .{ .gate = entry.gate.name, .revision = s.revision, .seconds = entry.result.seconds });
                    } else {
                        try tally.failed.append(s.arena, entry.gate.name);
                    }
                },
            }
            try s.results.append(s.arena, entry.result);
            try s.sayResult(entry);
            if (stopping and !s.options.keep_going and tally.failed.items.len > 0) stopped = true;
        }
        if (seconds.items.len > 0) {
            try s.times.save(s.arena, s.io, seconds.items);
            try s.passes.save(s.arena, s.io);
        }
    }

    fn writeJson(s: *Session) !void {
        const report = .{ .revision = s.revision, .builds = s.builds, .results = s.results.items };
        try std.json.Stringify.value(report, .{ .whitespace = .indent_2, .emit_null_optional_fields = false }, s.out);
        try s.out.writeByte('\n');
        try s.out.flush();
    }

    /// The closing line, and the exit status.
    fn finish(s: *Session, tally: *const Tally) !u8 {
        var total: f64 = 0;
        var ran: usize = 0;
        for (s.results.items) |r| {
            if (r.cached == null and r.skipped == null) {
                total += r.seconds;
                ran += 1;
            }
        }
        const failed = tally.failed.items;
        var line: Io.Writer.Allocating = .init(s.arena);
        const w = &line.writer;
        var failed_runs: usize = 0;
        for (s.results.items) |r| {
            if (r.passed != null and !r.passed.?) failed_runs += 1;
        }
        try w.print("{d}/{d} passed in {d:.0} s", .{ ran - failed_runs, ran, total });
        if (s.builds.any()) try w.print(" after {d:.0} s of builds", .{s.builds.total()});
        if (tally.cached > 0) try w.print("; {d} cached", .{tally.cached});
        if (tally.skipped.items.len > 0) {
            try w.print("; {d} skipped, their models or data not on this machine: {s}", .{ tally.skipped.items.len, try s.join(tally.skipped.items, ", ") });
        }
        if (failed.len > 0) try w.print("; failed: {s}", .{try s.join(failed, ", ")});
        try s.note("{s}", .{line.written()});
        if (s.options.json) try s.writeJson();
        return if (failed.len > 0) 1 else 0;
    }

    // ---- lanes --------------------------------------------------------------

    /// `--tier`, `--gate`: the gates in order, every one of them.
    pub fn runAll(s: *Session, gates: []const *const Gate) !u8 {
        var tally: Tally = .{};
        try s.execute(gates, &tally, false);
        if (s.options.dry_run) return 0;
        return s.finish(&tally);
    }

    /// `--sweep`: every tier's gates not already cached, cheapest first.
    pub fn sweep(s: *Session) !u8 {
        var all: std.ArrayList(*const Gate) = .empty;
        for (s.m.gates) |*gate| try all.append(s.arena, gate);
        var tally: Tally = .{};
        try s.execute(try select.byCost(s.arena, all.items, &s.times.times), &tally, true);
        if (s.options.dry_run) return 0;
        return s.finish(&tally);
    }

    /// `--changed`: the tier's gates the change selects, in manifest order.
    pub fn changed(s: *Session, rev: []const u8, tier: []const u8) !u8 {
        const change = try s.changeSince(rev);
        const selection = try select.impact(s.arena, s.m, change.hits, tier);
        try s.say("{d} changed file(s) since {s}; {d} engine file(s) changed; {d} {s} gate(s) selected: {s}", .{
            change.files.len,                                                           rev, change.hits.len, selection.gates.len, tier,
            if (selection.gates.len == 0) "none" else try s.gateNames(selection.gates),
        });
        try s.sayImpact(selection, tier);
        if (std.mem.eql(u8, tier, "verify")) try s.sayOtherTiers(selection);
        if (selection.gates.len == 0) return 0;
        return s.runAll(selection.gates);
    }

    fn sayOtherTiers(s: *Session, selection: select.Selection) !void {
        const Tier = struct { name: []const u8, label: []const u8 };
        const others = [_]Tier{
            .{ .name = "verify-long", .label = "Long-tier gates selected, not run (they run when the change touches attention, the caches, or a windowed schedule, and before a release; `--tier verify-long` runs them)" },
            .{ .name = "verify-cpu", .label = "CPU-tier gates selected, not run (they run when the change alters what the CPU reference computes, and before a release; `--tier verify-cpu` runs them)" },
            .{ .name = "verify-release", .label = "Release-tier gates selected, not run (before a release; `--tier verify-release` runs them)" },
        };
        for (others) |other| {
            var names: std.ArrayList(*const Gate) = .empty;
            for (s.m.gates) |*gate| {
                if (!std.mem.eql(u8, gate.tier, other.name)) continue;
                for (selection.reasons) |reason| {
                    if (std.mem.eql(u8, reason.family, gate.family)) try names.append(s.arena, gate);
                }
            }
            if (names.items.len > 0) try s.say("{s}: {s}", .{ other.label, try s.gateNames(names.items) });
        }
    }

    /// `--auto`, the Focus lane: the checks a change's paths select, the
    /// gates its impact selects cheapest first until one fails, then the
    /// tiers it requires but that did not run.
    pub fn auto(s: *Session, requested: ?[]const u8) !u8 {
        const base = requested orelse try git.unitBase(s.arena, s.io, null) orelse blk: {
            try s.note("warning: no REV and no origin/main; diffing against HEAD (committed work is not seen)", .{});
            break :blk "HEAD";
        };
        const change = try s.changeSince(base);
        try s.say("{d} changed file(s) since {s}; {d} engine file(s) changed", .{ change.files.len, base, change.hits.len });

        var tally: Tally = .{};
        const stop = struct {
            fn call(session: *const Session, t: *const Tally) bool {
                return t.failed.items.len > 0 and !session.options.keep_going;
            }
        }.call;

        // A deleted file still selects; only one that exists can be formatted.
        var present: std.ArrayList([]const u8) = .empty;
        for (change.files) |path| {
            if (exists(s.io, path)) try present.append(s.arena, path);
        }
        for (try select.matching(s.arena, manifest.Check, s.m.checks, change.files)) |check| {
            const argv = try select.checkArgv(s.arena, check, present.items) orelse continue;
            try s.say("check {s}: {s}", .{ check.name, try s.join(argv, " ") });
            if (s.options.dry_run) continue;
            try s.zigEnv();
            const started = Io.Clock.awake.now(s.io);
            const captured = capture(s.arena, s.gpa, s.io, s.env, argv) catch |err| {
                try s.say("FAIL check {s:<28} {s}", .{ check.name, @errorName(err) });
                try tally.failed.append(s.arena, check.name);
                if (stop(s, &tally)) break;
                continue;
            };
            const ok = captured.status == 0;
            try s.say("{s} check {s:<28} {d:>7.1} s", .{ if (ok) "PASS" else "FAIL", check.name, secondsSince(s.io, started) });
            if (ok) continue;
            try tally.failed.append(s.arena, check.name);
            for (captured.lines[captured.lines.len -| tail_lines..]) |line| try s.say("    {s}", .{line.text});
            if (stop(s, &tally)) break;
        }

        const selection = try select.impact(s.arena, s.m, change.hits, "verify");
        const gates = try select.byCost(s.arena, selection.gates, &s.times.times);
        if (gates.len > 0 and !stop(s, &tally)) {
            try s.sayImpact(selection, "verify");
            try s.say("{d} verify gate(s), cheapest first: {s}", .{ gates.len, try s.gateNames(gates) });
            try s.execute(gates, &tally, true);
        } else if (gates.len == 0) {
            try s.say("no gates selected", .{});
        }

        for (try select.requirements(s.arena, s.m.requires, change.effective)) |required| {
            try s.say("Required, not run: {s} ({s}: {s}); make verify-changed BASE={s} ARGS='--tier {s}'", .{ required.tier, required.file, required.reason, base, required.tier });
        }
        if (s.options.dry_run) return 0;
        if (tally.skipped.items.len > 0) try s.note("skipped, their models or data not on this machine: {s}", .{try s.join(tally.skipped.items, ", ")});
        if (tally.failed.items.len > 0) {
            try s.note("failed: {s}", .{try s.join(tally.failed.items, ", ")});
        } else {
            try s.note("all selected checks and gates passed ({d} cached, {d} skipped)", .{ tally.cached, tally.skipped.items.len });
        }
        if (s.options.json) try s.writeJson();
        return if (tally.failed.items.len > 0) 1 else 0;
    }

    /// `--replay FROM..TO`: what a committed range selects; runs nothing.
    pub fn replay(s: *Session, from: []const u8, to: []const u8) !u8 {
        const files = try git.changedBetween(s.arena, s.io, from, to) orelse {
            try s.note("git refuses {s}..{s}", .{ from, to });
            return error.BadRevision;
        };
        const change = try s.classify(files, from, to);
        const selection = try select.impact(s.arena, s.m, change.hits, "verify");
        const Output = struct {
            files: usize,
            engine: []const []const u8,
            families: []const []const u8,
            canaries: bool,
            gates: []const []const u8,
        };
        var engine: std.ArrayList([]const u8) = .empty;
        for (change.hits) |hit| try engine.append(s.arena, hit.file);
        var families: std.ArrayList([]const u8) = .empty;
        for (selection.reasons) |reason| try families.append(s.arena, reason.family);
        var names: std.ArrayList([]const u8) = .empty;
        for (selection.gates) |gate| try names.append(s.arena, gate.name);
        const output: Output = .{ .files = files.len, .engine = engine.items, .families = families.items, .canaries = selection.shared != null, .gates = names.items };
        if (s.options.json) {
            try std.json.Stringify.value(output, .{}, s.out);
            try s.out.writeByte('\n');
            try s.out.flush();
            return 0;
        }
        try s.say("{d} file(s), {d} engine file(s) changed, {d} gate(s): {s}", .{ files.len, engine.items.len, names.items.len, try s.join(names.items, ", ") });
        try s.sayImpact(selection, "verify");
        return 0;
    }

    /// `--validate`'s second half: what only the real tree can show.
    pub fn lint(s: *Session) !u8 {
        const problems = try tree_module.lint(s.arena, s.m, &s.tree, s.universe);
        for (problems) |problem| try s.note("gates.json: {s}", .{problem});
        return if (problems.len > 0) 2 else 0;
    }
};

fn secondsSince(io: Io, started: Io.Timestamp) f64 {
    const elapsed = started.durationTo(Io.Clock.awake.now(io)).toNanoseconds();
    return @as(f64, @floatFromInt(elapsed)) / std.time.ns_per_s;
}

fn tailText(arena: Allocator, lines: []const Line) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (lines, 0..) |line, i| {
        if (i > 0) try out.append(arena, '\n');
        try out.appendSlice(arena, line.text);
    }
    return out.items;
}

const Log = struct {
    gpa: Allocator,
    lines: std.ArrayList(Line) = .empty,

    fn add(log: *Log, t: f64, text: []const u8) Allocator.Error!void {
        const kept = try log.gpa.dupe(u8, text[0..@min(text.len, max_line_bytes)]);
        errdefer log.gpa.free(kept);
        if (log.lines.items.len == max_log_lines) log.gpa.free(log.lines.orderedRemove(0).text);
        try log.lines.append(log.gpa, .{ .t = t, .text = kept });
    }

    fn deinit(log: *Log) void {
        for (log.lines.items) |line| log.gpa.free(line.text);
        log.lines.deinit(log.gpa);
    }
};

const Captured = struct {
    /// The exit code, or minus the signal.
    status: i32,
    /// The newest lines, stamped with their arrival since launch.
    lines: []const Line,
};

/// Moves `r`'s complete lines into `log`, and with `final` the rest; an
/// unterminated line past the bound goes in as it stands.
fn drain(r: *Io.Reader, log: *Log, t: f64, final: bool) Allocator.Error!void {
    while (std.mem.indexOfScalar(u8, r.buffered(), '\n')) |end| {
        try log.add(t, std.mem.trimEnd(u8, r.buffered()[0..end], "\r"));
        r.toss(end + 1);
    }
    const rest = r.buffered();
    if (rest.len > 0 and (final or rest.len >= max_line_bytes)) {
        try log.add(t, rest);
        r.toss(rest.len);
    }
}

/// Runs `argv` with its stdout and stderr read line by line, each line stamped
/// on arrival: the phases of a check without instrumenting the tools. Output
/// is bounded to the newest `max_log_lines`; the result lives in `arena`.
fn capture(arena: Allocator, gpa: Allocator, io: Io, env: *const Environ.Map, argv: []const []const u8) !Captured {
    var child = try std.process.spawn(io, .{
        .argv = argv,
        .environ_map = env,
        .stdin = .ignore,
        .stdout = .pipe,
        .stderr = .pipe,
    });
    defer child.kill(io);
    var storage: Io.File.MultiReader.Buffer(2) = undefined;
    var reader: Io.File.MultiReader = undefined;
    reader.init(gpa, io, storage.toStreams(), &.{ child.stdout.?, child.stderr.? });
    defer reader.deinit();
    var log: Log = .{ .gpa = gpa };
    defer log.deinit();
    const started = Io.Clock.awake.now(io);
    while (true) {
        reader.fill(64, .none) catch |err| switch (err) {
            error.EndOfStream => break,
            else => |e| return e,
        };
        const t = secondsSince(io, started);
        try drain(reader.reader(0), &log, t, false);
        try drain(reader.reader(1), &log, t, false);
    }
    const t = secondsSince(io, started);
    try drain(reader.reader(0), &log, t, true);
    try drain(reader.reader(1), &log, t, true);
    try reader.checkAnyError();
    const term = try child.wait(io);
    const lines = try arena.alloc(Line, log.lines.items.len);
    for (lines, log.lines.items) |*slot, line| slot.* = .{ .t = line.t, .text = try arena.dupe(u8, line.text) };
    const status: i32 = switch (term) {
        .exited => |code| code,
        .signal, .stopped => |signal| -@as(i32, @intCast(@backingInt(signal))),
        .unknown => -1,
    };
    return .{ .status = status, .lines = lines };
}

const testing = std.testing;

fn sample(arena: Allocator) !Manifest {
    return manifest.load(arena, try manifest.parse(arena, manifest.sample));
}

test "placeholders expand, build ones to several arguments" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    const m = try sample(a);
    var env: Environ.Map = .init(a);
    try env.put("HOME", "/h");
    const cpu = try expand(a, &m, &m.gates[1], &env);
    try testing.expectEqualStrings("zig", cpu[0]);
    try testing.expectEqualStrings("build", cpu[1]);
    try testing.expectEqualStrings("-Doptimize=ReleaseFast", cpu[2]);
    try testing.expectEqualStrings("/h/m/a.gguf", cpu[cpu.len - 3]);
    try testing.expectEqualStrings("--draft-model", cpu[cpu.len - 2]);
    try testing.expectEqualStrings("/h/m/a-mtp.gguf", cpu[cpu.len - 1]);

    try env.put("A_MODEL", "/elsewhere/a.gguf");
    const trace = try expand(a, &m, &m.gates[0], &env);
    const expected = [_][]const u8{ "./zig-out/bin/nuclis", "generate", "--model", "/elsewhere/a.gguf", "--trace-dir", trace_root ++ "/a-trace" };
    try testing.expectEqual(expected.len, trace.len);
    for (expected, trace) |want, got| try testing.expectEqualStrings(want, got);
}

test "a gate lists every model file it reads, overrides included" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    const m = try sample(a);
    var env: Environ.Map = .init(a);
    try env.put("A_MODEL", "/nowhere/a.gguf");
    try env.put("A_MTP_MODEL", "/nowhere/mtp.gguf");
    const files = try modelFiles(a, &m, &m.gates[1], &env);
    try testing.expectEqual(@as(usize, 2), files.len);
    try testing.expectEqualStrings("/nowhere/a.gguf", files[0]);
    try testing.expectEqualStrings("/nowhere/mtp.gguf", files[1]);
    try testing.expect(!exists(testing.io, files[0]));
}

test "an {entry} gate runs the catalogue name and keys on the pinned files" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    var m = try sample(a);
    const models = try a.dupe(manifest.Model, m.models);
    models[0].entry = "a-entry";
    models[0].mmproj = "~/m/a-mmproj.gguf";
    m.models = models;
    var gate = m.gates[0];
    gate.command = &.{ "{nuclis}", "generate", "--model", "{entry}" };
    var env: Environ.Map = .init(a);
    try env.put("HOME", "/h");
    try env.put("A_MODEL", "/elsewhere/a.gguf");
    const argv = try expand(a, &m, &gate, &env);
    try testing.expectEqualStrings("a-entry", argv[3]);
    const files = try modelFiles(a, &m, &gate, &env);
    try testing.expectEqual(@as(usize, 2), files.len);
    try testing.expectEqualStrings("/h/m/a.gguf", files[0]);
    try testing.expectEqualStrings("/h/m/a-mmproj.gguf", files[1]);
}

test "missing data names the set and its fetch" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    const m = try sample(a);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try a.print(".zig-cache/tmp/{s}", .{tmp.sub_path});
    var gate = m.gates[0];
    gate.data = &.{ "text", "clips" };
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "corpus.json", .data = "{\"clips\": [{\"file\": \"a.flac\"}, {\"file\": \"b/c.wav\"}]}" });
    var missing = try missingData(a, testing.io, &m, &gate, root);
    try testing.expectEqual(@as(usize, 2), missing.len);
    try testing.expectEqualStrings("text", missing[0].name);
    try testing.expectEqual(@as(usize, 1), missing[0].total);
    try testing.expectEqualStrings("make eval-corpus", missing[0].fetch);
    try testing.expectEqual(@as(usize, 2), missing[1].absent);
    try testing.expectEqualStrings("make speech-corpus", missing[1].fetch);

    try tmp.dir.createDirPath(testing.io, "eval");
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "eval/t.raw", .data = "x" });
    try tmp.dir.createDirPath(testing.io, "speech/b");
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "speech/a.flac", .data = "x" });
    missing = try missingData(a, testing.io, &m, &gate, root);
    try testing.expectEqual(@as(usize, 1), missing.len);
    try testing.expectEqual(@as(usize, 1), missing[0].absent);
    try testing.expectEqual(@as(usize, 2), missing[0].total);

    try tmp.dir.writeFile(testing.io, .{ .sub_path = "speech/b/c.wav", .data = "x" });
    try testing.expectEqual(@as(usize, 0), (try missingData(a, testing.io, &m, &gate, root)).len);
    try testing.expectEqual(@as(usize, 0), (try missingData(a, testing.io, &m, &m.gates[0], root)).len);
}

test "capture stamps both streams and reports the exit status" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    var env: Environ.Map = .init(a);
    try env.put("PATH", "/usr/bin:/bin");
    const captured = try capture(a, testing.allocator, testing.io, &env, &.{ "/bin/sh", "-c", "echo out; echo err >&2; printf tail; exit 3" });
    try testing.expectEqual(@as(i32, 3), captured.status);
    try testing.expectEqual(@as(usize, 3), captured.lines.len);
    var seen_err = false;
    for (captured.lines) |line| {
        if (std.mem.eql(u8, line.text, "err")) seen_err = true;
    }
    try testing.expect(seen_err);
    const signalled = try capture(a, testing.allocator, testing.io, &env, &.{ "/bin/sh", "-c", "kill -9 $$" });
    try testing.expectEqual(@as(i32, -9), signalled.status);
}

test "the log keeps the newest lines and frees the rest" {
    var log: Log = .{ .gpa = testing.allocator };
    defer log.deinit();
    for (0..max_log_lines + 3) |i| {
        var buffer: [32]u8 = undefined;
        try log.add(@floatFromInt(i), try std.fmt.bufPrint(&buffer, "line {d}", .{i}));
    }
    try testing.expectEqual(@as(usize, max_log_lines), log.lines.items.len);
    try testing.expectEqualStrings("line 3", log.lines.items[0].text);
}
