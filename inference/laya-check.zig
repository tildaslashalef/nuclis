//! Explicit full-model check of Laya's CPU forward against the oracle's
//! fixtures (scripts/laya-reference.py): every request of `requests.json`
//! runs through the encoder and head, and each stage the fixture holds is
//! compared row by row with `activations.f32`, then the logits. Needs the
//! pulled checkpoint directory; no GPU.
const std = @import("std");
const inference = @import("inference");
const laya = inference.models.laya;

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
    if (args.len != 2) return error.ExpectedLayaDirectory;
    const parsed = try std.json.parseFromSlice(struct { hidden: usize, requests: []const Request }, gpa, requests_json, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();

    const load_start = std.Io.Clock.awake.now(io);
    var model = try laya.Laya.open(gpa, io, args[1]);
    defer model.deinit(io);
    const load_ms = load_start.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();
    if (model.config.hidden != parsed.value.hidden) return error.ArtifactMismatch;
    std.debug.print("Laya loaded in {d} ms: {d} encoder layers, {d} head layers.\n", .{ load_ms, model.config.layers, model.head.layers.len });

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
    std.debug.print("Laya CPU check passed.\n", .{});
}
