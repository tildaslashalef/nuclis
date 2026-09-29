//! Explicit full-model check of Laya's forward against the oracle's fixtures
//! (scripts/laya-reference.py): every request of `requests.json` runs
//! through the encoder and head, and each stage the fixture holds is
//! compared row by row with `activations.f32`, then the logits. Needs the
//! pulled checkpoint directory. `--backend metal` checks the Metal plan the
//! same way, then that one packed batch of every request gives exactly the
//! logits of one at a time, and the plan's cleanup on the tiny checkpoint;
//! `--profile` after it prints the GPU time per kernel of the 512-token
//! request and of all 8 packed.
const std = @import("std");
const inference = @import("inference");
const laya = inference.models.laya;
const metal = inference.metal;

const requests_json = @embedFile("src/models/fixtures/laya/requests.json");
const activations = @embedFile("src/models/fixtures/laya/activations.f32");

/// Bounds relative to magnitude, since F32 summation-order differences scale
/// with the values summed and the pre-norm residual carries outliers near
/// 1e4: per tensor, the largest difference over the largest reference value,
/// and the difference's RMS over the reference's; per logit, the difference
/// over max(1, |logit|).
const max_bound = 1e-5;
const rms_bound = 1e-5;
const logit_bound = 1e-4;

const Tensor = struct { name: []const u8, rows: []const usize, offset: usize };
const Request = struct { name: []const u8, qtype: u2, ids: []const u32, markers: []const usize, logits: []const f32, tensors: []const Tensor };

const Stats = struct { max_abs: f64 = 0, max_ref: f64 = 0, diff2: f64 = 0, ref2: f64 = 0, seen: bool = false };

const Compare = struct {
    request: *const Request,
    hidden: usize,
    stats: []Stats,

    fn record(context: *anyopaque, stage: laya.Stage, states: []const f32) void {
        const self: *Compare = @ptrCast(@alignCast(context));
        var buffer: [32]u8 = undefined;
        const name = switch (stage) {
            .encoder => |i| std.fmt.bufPrint(&buffer, "encoder.{d}", .{i}) catch unreachable,
            .final => "final",
            .head => |i| std.fmt.bufPrint(&buffer, "head.{d}", .{i}) catch unreachable,
        };
        for (self.request.tensors, self.stats) |t, *s| {
            if (!std.mem.eql(u8, t.name, name)) continue;
            s.seen = true;
            for (t.rows, 0..) |row, r| for (0..self.hidden) |c| {
                const want: f64 = reference(t.offset + r * self.hidden + c);
                const d = states[row * self.hidden + c] - want;
                s.max_abs = @max(s.max_abs, @abs(d));
                s.max_ref = @max(s.max_ref, @abs(want));
                s.diff2 += d * d;
                s.ref2 += want * want;
            };
        }
    }
};

fn reference(index: usize) f32 {
    return @bitCast(std.mem.readInt(u32, activations[index * 4 ..][0..4], .little));
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2 and !(args.len >= 4 and args.len <= 5 and std.mem.eql(u8, args[2], "--backend"))) return error.ExpectedLayaDirectory;
    const backend: laya.Backend = if (args.len >= 4) std.meta.stringToEnum(laya.Backend, args[3]) orelse return error.UnknownBackend else .cpu;
    const profile = args.len == 5 and std.mem.eql(u8, args[4], "--profile") and backend == .metal;
    if (args.len == 5 and !profile) return error.UnknownOption;
    const parsed = try std.json.parseFromSlice(struct { hidden: usize, requests: []const Request }, gpa, requests_json, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();

    const load_start = std.Io.Clock.awake.now(io);
    var model = try laya.Laya.open(gpa, io, args[1], backend);
    defer model.deinit(io);
    const load_ms = load_start.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();
    if (model.config.hidden != parsed.value.hidden) return error.ArtifactMismatch;
    std.debug.print("Laya loaded on the {s} in {d} ms: {d} encoder layers, {d} head layers.\n", .{ @tagName(backend), load_ms, model.config.layers, model.head_layers });

    var worst_max: f64 = 0;
    var worst_rms: f64 = 0;
    var worst_logit: f64 = 0;
    var failed = false;
    for (parsed.value.requests) |*r| {
        const stats = try gpa.alloc(Stats, r.tensors.len);
        defer gpa.free(stats);
        @memset(stats, .{});
        var compare: Compare = .{ .request = r, .hidden = parsed.value.hidden, .stats = stats };
        const logits = try gpa.alloc(f32, r.markers.len);
        defer gpa.free(logits);
        const start = std.Io.Clock.awake.now(io);
        try model.logits(io, gpa, r.ids, r.markers, @enumFromInt(r.qtype), logits, .{ .context = &compare, .record = Compare.record });
        const ms = start.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();
        var logit_error: f64 = 0;
        for (logits, r.logits) |got, want| logit_error = @max(logit_error, @abs(@as(f64, got) - want) / @max(1, @abs(want)));
        worst_logit = @max(worst_logit, logit_error);
        std.debug.print("{s}: {d} tokens, {d} options, {d} ms; logits scaled |Δ| {e:.2}\n", .{ r.name, r.ids.len, r.markers.len, ms, logit_error });
        if (logit_error > logit_bound) failed = true;
        for (r.tensors, stats) |t, s| {
            if (!s.seen) return error.StageNotTraced;
            const scaled = s.max_abs / s.max_ref;
            const rms = @sqrt(s.diff2 / s.ref2);
            worst_max = @max(worst_max, scaled);
            worst_rms = @max(worst_rms, rms);
            const bad = scaled > max_bound or rms > rms_bound;
            if (bad) failed = true;
            std.debug.print("  {s:<10} max |Δ| {e:.2} of max |ref| {e:.2} ({e:.2})  rel RMS {e:.2}{s}\n", .{ t.name, s.max_abs, s.max_ref, scaled, rms, if (bad) "  OVER" else "" });
        }
    }
    std.debug.print("Worst over {d} requests: scaled max |Δ| {e:.2} (bound {e:.0}), rel RMS {e:.2} (bound {e:.0}), scaled logit |Δ| {e:.2} (bound {e:.0}).\n", .{ parsed.value.requests.len, worst_max, max_bound, worst_rms, rms_bound, worst_logit, logit_bound });
    if (failed) return error.OutsideTolerance;
    if (backend == .metal) {
        try checkBatch(gpa, io, &model, parsed.value.requests);
        try checkLifecycle(gpa, io);
        if (profile) try profileKernels(gpa, io, &model, parsed.value.requests);
    }
    std.debug.print("Laya {s} check passed.\n", .{@tagName(backend)});
}

/// Every request in one `logitsBatch` call against each run alone: a row's
/// arithmetic does not depend on what else is packed beside it, so the
/// logits must be equal to the bit.
fn checkBatch(gpa: std.mem.Allocator, io: std.Io, model: *const laya.Laya, requests: []const Request) !void {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const sequences = try a.alloc(laya.Sequence, requests.len);
    const batched = try a.alloc([]f32, requests.len);
    var rows: usize = 0;
    for (requests, sequences, batched) |r, *s, *out| {
        s.* = .{ .ids = r.ids, .markers = r.markers, .kind = @enumFromInt(r.qtype) };
        out.* = try a.alloc(f32, r.markers.len);
        rows += r.ids.len;
    }
    const start = std.Io.Clock.awake.now(io);
    try model.logitsBatch(io, a, sequences, batched);
    const ms = start.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();
    for (sequences, batched) |s, got| {
        const alone = try a.alloc(f32, s.markers.len);
        try model.logits(io, a, s.ids, s.markers, s.kind, alone, null);
        if (!std.mem.eql(u32, @ptrCast(alone), @ptrCast(got))) return error.BatchMismatch;
    }
    std.debug.print("One packed batch of all {d} requests ({d} rows, {d} ms): logits bit-identical to one at a time.\n", .{ requests.len, rows, ms });
}

/// The plan on the tiny checkpoint: repeated open and close, a forward, and
/// every allocation of `open` failing in turn without a leak.
fn checkLifecycle(gpa: std.mem.Allocator, io: std.Io) !void {
    var tmp_path_buffer: [64]u8 = undefined;
    const tmp = try std.fmt.bufPrint(&tmp_path_buffer, ".zig-cache/tmp/laya-tiny-{d}", .{std.Io.Clock.real.now(io).toNanoseconds()});
    const cwd = std.Io.Dir.cwd();
    try cwd.createDirPath(io, tmp);
    defer cwd.deleteTree(io, tmp) catch {};
    var dir = try cwd.openDir(io, tmp, .{});
    defer dir.close(io);
    try laya.writeTiny(gpa, io, dir, null, null);
    for (0..3) |_| {
        var model = try laya.Laya.open(gpa, io, tmp, .metal);
        defer model.deinit(io);
        var out: [2]f32 = undefined;
        try model.logits(io, gpa, &.{ 1, 5, 19, 2, 7 }, &.{ 1, 3 }, .noul, &out, null);
        if (out[0] != 0.25 or out[1] != 0.25) return error.TinyLogits;
    }
    const Open = struct {
        fn run(allocator: std.mem.Allocator, io_: std.Io, path: []const u8) !void {
            var model = try laya.Laya.open(allocator, io_, path, .metal);
            model.deinit(io_);
        }
    };
    try std.testing.checkAllAllocationFailures(gpa, Open.run, .{ io, tmp });
    std.debug.print("Tiny checkpoint on Metal: three open/forward/close cycles; every allocation of open failing in turn, no leak.\n", .{});
}

/// GPU time per kernel and shape, each dispatch timestamped (which adds its
/// own encoder boundaries): the 512-token `long_text` request alone, then
/// all 8 requests packed, each after a warm-up run.
fn profileKernels(gpa: std.mem.Allocator, io: std.Io, model: *const laya.Laya, requests: []const Request) !void {
    const backend = &model.engine.metal.backend;
    var diagnostic: [256]u8 = undefined;
    try backend.enableProfiling(4096, &diagnostic);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const sequences = try a.alloc(laya.Sequence, requests.len);
    const outs = try a.alloc([]f32, requests.len);
    var long: usize = 0;
    for (requests, sequences, outs, 0..) |r, *s, *out, i| {
        s.* = .{ .ids = r.ids, .markers = r.markers, .kind = @enumFromInt(r.qtype) };
        out.* = try a.alloc(f32, r.markers.len);
        if (std.mem.eql(u8, r.name, "long_text")) long = i;
    }
    for ([_][]const laya.Sequence{ sequences[long..][0..1], sequences }, [_][]const u8{ "long_text alone", "all 8 packed" }) |batch, label| {
        try model.logitsBatch(io, a, batch, outs[0..batch.len]);
        backend.profile.?.clear();
        try model.logitsBatch(io, a, batch, outs[0..batch.len]);
        const p = &backend.profile.?;
        const Row = struct { key: metal.Profile.Key, total: metal.Profile.Total };
        var rows: std.ArrayList(Row) = .empty;
        defer rows.deinit(gpa);
        var sum: f64 = 0;
        var it = p.totals.iterator();
        while (it.next()) |e| {
            try rows.append(gpa, .{ .key = e.key_ptr.*, .total = e.value_ptr.* });
            sum += e.value_ptr.seconds;
        }
        std.mem.sort(Row, rows.items, {}, struct {
            fn lessThan(_: void, x: Row, y: Row) bool {
                return x.total.seconds > y.total.seconds;
            }
        }.lessThan);
        var rows_total: usize = 0;
        for (batch) |s| rows_total += s.ids.len;
        std.debug.print("Profile, {s} ({d} rows): {d:.1} ms in kernels, {d:.1} ms GPU, {d} unsampled\n", .{ label, rows_total, sum * 1e3, p.gpu_seconds * 1e3, p.unsampled });
        for (rows.items) |r| std.debug.print("  {s:<22} {d:>5}x{d:<5} {d:>4} dispatches {d:>8.2} ms {d:>5.1}%\n", .{ @tagName(r.key.kernel), r.key.rows, r.key.columns, r.total.dispatches, r.total.seconds * 1e3, 100 * r.total.seconds / sum });
    }
}
