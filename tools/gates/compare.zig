//! The trace comparator: a run's F32 trace files against a reference
//! directory, within absolute and relative-RMS bounds. Layer mode reads each
//! position's layer residuals and `logits.f32`; draft mode the prediction-block
//! rows and `greedy.txt`. Used by `trace` gates and by `nuclis-gates compare`.
const std = @import("std");

pub const Options = struct {
    positions: u32,
    max_absolute: f64 = 0.002,
    max_relative_rms: f64 = 0.0001,
    /// Residual width per layer file; required. The shapes have no default:
    /// they come from the gate (docs/development.md § Gates).
    embedding: u32 = 0,
    /// Decoder layers traced; required in layer mode.
    layers: u32 = 0,
    /// Logit count; required in layer mode.
    vocab: u32 = 0,
    /// Compare `p0-h.f32`, `p1-h.f32`, `p1-hprev.f32`, and `greedy.txt`.
    draft: bool = false,
};

pub const max_positions = 32768;
const max_greedy_bytes = 1 << 20;

pub const Row = struct {
    file: []const u8,
    max_absolute: f64,
    rms: f64,
    relative_rms: f64,
    native_top5: ?[]const u32 = null,
    reference_top5: ?[]const u32 = null,
    reference_greedy_margin: ?f64 = null,
};

/// Field order is the report's JSON order; absent draft fields are omitted.
pub const Report = struct {
    passed: bool,
    max_absolute_tolerance: f64,
    relative_rms_tolerance: f64,
    native_greedy: ?[]const i64 = null,
    reference_greedy: ?[]const i64 = null,
    comparisons: []const Row,

    pub fn maxAbsolute(report: Report) f64 {
        var worst: f64 = 0;
        for (report.comparisons) |row| worst = @max(worst, row.max_absolute);
        return worst;
    }

    pub fn maxRelativeRms(report: Report) f64 {
        var worst: f64 = 0;
        for (report.comparisons) |row| worst = @max(worst, row.relative_rms);
        return worst;
    }

    pub fn write(report: Report, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try std.json.Stringify.value(report, .{ .whitespace = .indent_2, .emit_null_optional_fields = false }, w);
        try w.writeByte('\n');
    }
};

pub const Error = error{
    InvalidOptions,
    MissingFile,
    UnexpectedSize,
    Nonfinite,
    NoDraftRows,
    BadGreedy,
    ReadFailed,
    OutOfMemory,
};

/// What `Error` was about: the file, for the report.
pub const Diagnostic = struct {
    path: []const u8 = "",
};

/// Compares `native` against `reference`. The report and its strings live in
/// `arena`; each file's samples are read through `gpa` and freed.
pub fn compare(
    arena: std.mem.Allocator,
    gpa: std.mem.Allocator,
    io: std.Io,
    native: []const u8,
    reference: []const u8,
    options: Options,
    diag: *Diagnostic,
) Error!Report {
    if (options.positions < 1 or options.positions > max_positions) return error.InvalidOptions;
    if (options.embedding == 0) return error.InvalidOptions;
    if (!options.draft and (options.layers == 0 or options.vocab < 2)) return error.InvalidOptions;
    var rows: std.ArrayList(Row) = .empty;
    if (options.draft) {
        for (0..options.positions) |p| {
            const name = try arena.print("p{d}-h.f32", .{p});
            try appendOptional(arena, gpa, io, &rows, native, reference, name, options.embedding, diag);
        }
        try appendOptional(arena, gpa, io, &rows, native, reference, "p1-hprev.f32", options.embedding, diag);
        if (rows.items.len == 0) return error.NoDraftRows;
        const native_greedy = try readGreedy(arena, io, native, diag);
        const reference_greedy = try readGreedy(arena, io, reference, diag);
        return .{
            .passed = within(rows.items, options) and std.mem.eql(i64, native_greedy, reference_greedy),
            .max_absolute_tolerance = options.max_absolute,
            .relative_rms_tolerance = options.max_relative_rms,
            .native_greedy = native_greedy,
            .reference_greedy = reference_greedy,
            .comparisons = rows.items,
        };
    }
    for (0..options.positions) |p| {
        for (0..options.layers) |layer| {
            const name = try arena.print("token-{d}-layer-{d}.f32", .{ p, layer });
            const a = try readSamples(arena, gpa, io, native, name, options.embedding, diag);
            defer gpa.free(a);
            const b = try readSamples(arena, gpa, io, reference, name, options.embedding, diag);
            defer gpa.free(b);
            try rows.append(arena, metric(name, a, b));
        }
    }
    const a = try readSamples(arena, gpa, io, native, "logits.f32", options.vocab, diag);
    defer gpa.free(a);
    const b = try readSamples(arena, gpa, io, reference, "logits.f32", options.vocab, diag);
    defer gpa.free(b);
    var logits = metric("logits.f32", a, b);
    logits.native_top5 = try arena.dupe(u32, top5(a).slice());
    logits.reference_top5 = try arena.dupe(u32, top5(b).slice());
    logits.reference_greedy_margin = b[logits.reference_top5.?[0]] - b[logits.reference_top5.?[1]];
    try rows.append(arena, logits);
    return .{
        .passed = within(rows.items, options),
        .max_absolute_tolerance = options.max_absolute,
        .relative_rms_tolerance = options.max_relative_rms,
        .comparisons = rows.items,
    };
}

fn within(rows: []const Row, options: Options) bool {
    for (rows) |row| {
        if (!(row.max_absolute <= options.max_absolute and row.relative_rms <= options.max_relative_rms)) return false;
    }
    return true;
}

fn appendOptional(
    arena: std.mem.Allocator,
    gpa: std.mem.Allocator,
    io: std.Io,
    rows: *std.ArrayList(Row),
    native: []const u8,
    reference: []const u8,
    name: []const u8,
    size: u32,
    diag: *Diagnostic,
) Error!void {
    // Optional means the reference may lack the row (`dflash2` has no
    // `p1-hprev`); a row the reference has must exist in the run.
    const b = readSamples(arena, gpa, io, reference, name, size, diag) catch |err| switch (err) {
        error.MissingFile => return,
        else => |e| return e,
    };
    defer gpa.free(b);
    const a = try readSamples(arena, gpa, io, native, name, size, diag);
    defer gpa.free(a);
    try rows.append(arena, metric(name, a, b));
}

/// One file's `size` little-endian F32 samples, owned by `gpa`; refuses the
/// wrong length and any non-finite value.
fn readSamples(
    arena: std.mem.Allocator,
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    name: []const u8,
    size: u32,
    diag: *Diagnostic,
) Error![]f32 {
    const path = try std.fs.path.join(arena, &.{ dir, name });
    diag.path = path;
    const expected = @as(usize, size) * 4;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(expected + 1)) catch |err| switch (err) {
        error.FileNotFound => return error.MissingFile,
        error.StreamTooLong => return error.UnexpectedSize,
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.ReadFailed,
    };
    defer gpa.free(bytes);
    if (bytes.len != expected) return error.UnexpectedSize;
    const samples = try gpa.alloc(f32, size);
    errdefer gpa.free(samples);
    for (samples, 0..) |*sample, i| {
        sample.* = @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
        if (!std.math.isFinite(sample.*)) return error.Nonfinite;
    }
    return samples;
}

fn metric(name: []const u8, a: []const f32, b: []const f32) Row {
    var sum: f64 = 0;
    var baseline: f64 = 0;
    var worst: f64 = 0;
    for (a, b) |x, y| {
        const delta = @as(f64, x) - @as(f64, y);
        sum += delta * delta;
        baseline += @as(f64, y) * @as(f64, y);
        worst = @max(worst, @abs(delta));
    }
    const size: f64 = @floatFromInt(a.len);
    const rms = @sqrt(sum / size);
    return .{
        .file = name,
        .max_absolute = worst,
        .rms = rms,
        .relative_rms = rms / @max(@sqrt(baseline / size), 1e-30),
    };
}

const Top = struct {
    indices: [5]u32 = undefined,
    len: usize = 0,

    fn slice(top: *const Top) []const u32 {
        return top.indices[0..top.len];
    }
};

/// The five largest values' indices, largest first; ties keep the lower index.
fn top5(values: []const f32) Top {
    var top: Top = .{};
    for (values, 0..) |value, i| {
        var slot = top.len;
        while (slot > 0 and values[top.indices[slot - 1]] < value) slot -= 1;
        if (slot == top.indices.len) continue;
        if (top.len < top.indices.len) top.len += 1;
        std.mem.copyBackwards(u32, top.indices[slot + 1 .. top.len], top.indices[slot .. top.len - 1]);
        top.indices[slot] = @intCast(i);
    }
    return top;
}

fn readGreedy(arena: std.mem.Allocator, io: std.Io, dir: []const u8, diag: *Diagnostic) Error![]const i64 {
    const path = try std.fs.path.join(arena, &.{ dir, "greedy.txt" });
    diag.path = path;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(max_greedy_bytes)) catch |err| switch (err) {
        error.FileNotFound => return error.MissingFile,
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.ReadFailed,
    };
    var values: std.ArrayList(i64) = .empty;
    var words = std.mem.tokenizeAny(u8, bytes, " \t\r\n");
    while (words.next()) |word| {
        try values.append(arena, std.fmt.parseInt(i64, word, 10) catch return error.BadGreedy);
    }
    return values.items;
}

const testing = std.testing;

fn writeF32(dir: std.Io.Dir, name: []const u8, values: []const f32) !void {
    try dir.writeFile(testing.io, .{ .sub_path = name, .data = std.mem.sliceAsBytes(values) });
}

const Fixture = struct {
    tmp: testing.TmpDir,
    arena: std.heap.ArenaAllocator,
    native: []const u8,
    reference: []const u8,

    fn init() !Fixture {
        var tmp = testing.tmpDir(.{});
        errdefer tmp.cleanup();
        var arena: std.heap.ArenaAllocator = .init(testing.allocator);
        errdefer arena.deinit();
        const a = arena.allocator();
        try tmp.dir.createDir(testing.io, "n", .default_dir);
        try tmp.dir.createDir(testing.io, "r", .default_dir);
        const base = try a.print(".zig-cache/tmp/{s}", .{tmp.sub_path});
        return .{
            .tmp = tmp,
            .arena = arena,
            .native = try std.fs.path.join(a, &.{ base, "n" }),
            .reference = try std.fs.path.join(a, &.{ base, "r" }),
        };
    }

    fn deinit(f: *Fixture) void {
        f.arena.deinit();
        f.tmp.cleanup();
    }

    fn dir(f: *Fixture, sub: []const u8) !std.Io.Dir {
        return f.tmp.dir.openDir(testing.io, sub, .{});
    }

    fn write(f: *Fixture, sub: []const u8, name: []const u8, values: []const f32) !void {
        const d = try f.dir(sub);
        defer d.close(testing.io);
        try writeF32(d, name, values);
    }
};

test "layer mode reports the logits' top five and the reference margin" {
    var f: Fixture = try .init();
    defer f.deinit();
    const a = f.arena.allocator();
    for ([_][]const u8{ "n", "r" }) |sub| {
        try f.write(sub, "token-0-layer-0.f32", &.{ 1, 2 });
        try f.write(sub, "token-0-layer-1.f32", &.{ 3, 4 });
    }
    try f.write("n", "logits.f32", &.{ 0.5, 3, 2, 2, 1, -1 });
    try f.write("r", "logits.f32", &.{ 0.5, 3, 2.5, 2, 1, -1 });
    var diag: Diagnostic = .{};
    const report = try compare(a, testing.allocator, testing.io, f.native, f.reference, .{ .positions = 1, .embedding = 2, .layers = 2, .vocab = 6, .max_absolute = 0.6, .max_relative_rms = 1 }, &diag);
    try testing.expect(report.passed);
    try testing.expectEqual(@as(usize, 3), report.comparisons.len);
    const logits = report.comparisons[2];
    try testing.expectEqualStrings("logits.f32", logits.file);
    try testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 0 }, logits.native_top5.?);
    try testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 0 }, logits.reference_top5.?);
    try testing.expectApproxEqAbs(@as(f64, 0.5), logits.reference_greedy_margin.?, 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.5), logits.max_absolute, 1e-9);
    try testing.expectApproxEqAbs(@sqrt(0.25 / 6.0), logits.rms, 1e-9);
}

test "a value past either bound fails the report" {
    var f: Fixture = try .init();
    defer f.deinit();
    const a = f.arena.allocator();
    try f.write("n", "token-0-layer-0.f32", &.{ 1, 2 });
    try f.write("r", "token-0-layer-0.f32", &.{ 1, 2.5 });
    try f.write("n", "logits.f32", &.{ 1, 0 });
    try f.write("r", "logits.f32", &.{ 1, 0 });
    var diag: Diagnostic = .{};
    const options: Options = .{ .positions = 1, .embedding = 2, .layers = 1, .vocab = 2 };
    const report = try compare(a, testing.allocator, testing.io, f.native, f.reference, options, &diag);
    try testing.expect(!report.passed);
    try testing.expectApproxEqAbs(@as(f64, 0.5), report.maxAbsolute(), 1e-9);
}

test "draft mode skips rows the reference lacks and compares the greedy tokens" {
    var f: Fixture = try .init();
    defer f.deinit();
    const a = f.arena.allocator();
    for ([_][]const u8{ "n", "r" }) |sub| {
        try f.write(sub, "p0-h.f32", &.{ 1, 2 });
        const d = try f.dir(sub);
        defer d.close(testing.io);
        try d.writeFile(testing.io, .{ .sub_path = "greedy.txt", .data = "5 7\n9\n" });
    }
    var diag: Diagnostic = .{};
    const options: Options = .{ .positions = 2, .embedding = 2, .draft = true };
    var report = try compare(a, testing.allocator, testing.io, f.native, f.reference, options, &diag);
    try testing.expect(report.passed);
    try testing.expectEqual(@as(usize, 1), report.comparisons.len);
    try testing.expectEqualSlices(i64, &.{ 5, 7, 9 }, report.native_greedy.?);

    const d = try f.dir("n");
    defer d.close(testing.io);
    try d.writeFile(testing.io, .{ .sub_path = "greedy.txt", .data = "5 7 8\n" });
    report = try compare(a, testing.allocator, testing.io, f.native, f.reference, options, &diag);
    try testing.expect(!report.passed);

    // A row only the reference has is a run that stopped writing it.
    try f.write("r", "p1-h.f32", &.{ 3, 4 });
    try testing.expectError(error.MissingFile, compare(a, testing.allocator, testing.io, f.native, f.reference, options, &diag));
    try testing.expect(std.mem.endsWith(u8, diag.path, "n/p1-h.f32"));
}

test "wrong sizes, non-finite values, and missing files are typed" {
    var f: Fixture = try .init();
    defer f.deinit();
    const a = f.arena.allocator();
    var diag: Diagnostic = .{};
    const options: Options = .{ .positions = 1, .embedding = 2, .layers = 1, .vocab = 2 };
    try testing.expectError(error.MissingFile, compare(a, testing.allocator, testing.io, f.native, f.reference, options, &diag));
    try f.write("n", "token-0-layer-0.f32", &.{ 1, 2, 3 });
    try testing.expectError(error.UnexpectedSize, compare(a, testing.allocator, testing.io, f.native, f.reference, options, &diag));
    try f.write("n", "token-0-layer-0.f32", &.{ 1, std.math.inf(f32) });
    try testing.expectError(error.Nonfinite, compare(a, testing.allocator, testing.io, f.native, f.reference, options, &diag));
    try testing.expectError(error.InvalidOptions, compare(a, testing.allocator, testing.io, f.native, f.reference, .{ .positions = 0 }, &diag));
    try testing.expectError(error.NoDraftRows, compare(a, testing.allocator, testing.io, f.native, f.reference, .{ .positions = 1, .embedding = 2, .draft = true }, &diag));
    try testing.expectError(error.InvalidOptions, compare(a, testing.allocator, testing.io, f.native, f.reference, .{ .positions = 1, .draft = true }, &diag));
    try testing.expectError(error.InvalidOptions, compare(a, testing.allocator, testing.io, f.native, f.reference, .{ .positions = 1, .embedding = 2, .vocab = 2 }, &diag));
}

test "top five keeps the lower index on a tie and the report's JSON omits absent fields" {
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 4 }, top5(&.{ 1, 1, 1, 1, 1, 1 }).slice());
    try testing.expectEqualSlices(u32, &.{ 1, 0 }, top5(&.{ 1, 2 }).slice());
    var buffer: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buffer);
    const report: Report = .{ .passed = true, .max_absolute_tolerance = 0.5, .relative_rms_tolerance = 0.25, .comparisons = &.{} };
    try report.write(&w);
    try testing.expect(std.mem.indexOf(u8, w.buffered(), "native_greedy") == null);
    try testing.expect(std.mem.indexOf(u8, w.buffered(), "\"passed\": true") != null);
}
