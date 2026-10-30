//! drama3-check's `codec-metal` mode: Fish's pinned codes (`codec-reference`,
//! `codec-window` under `tests/fixtures/drama3/`) decoded by
//! `fish_codec_metal` against the CPU decoder (`fish_codec`) on the same
//! codes. The Metal stream runs in pushes of 1 and 4 frames and the whole;
//! every stage the decoders observe is held to a relative RMS bound over all
//! its rows, and the three runs must give the same samples bit for bit.
//! Then each push size is timed unobserved.
const std = @import("std");
const inference = @import("inference");
const fc = inference.audio.fish_codec;
const fm = inference.audio.fish_codec_metal;

const stages = std.enums.values(fc.Stage);
const pushes = [_]usize{ 1, 4, 0 };

/// Every stage's rows, concatenated over the pushes.
const Recorder = struct {
    gpa: std.mem.Allocator,
    rows: [stages.len]std.ArrayList(f32) = @splat(.empty),

    fn record(context: *anyopaque, which: fc.Stage, rows: []const f32) void {
        const self: *Recorder = @ptrCast(@alignCast(context));
        self.rows[@backingInt(which)].appendSlice(self.gpa, rows) catch @panic("out of memory");
    }
    fn deinit(self: *Recorder) void {
        for (&self.rows) |*r| r.deinit(self.gpa);
    }
};

/// Each stage's rows against a recorder's, as they come.
const Comparer = struct {
    want: *const Recorder,
    at: [stages.len]usize = @splat(0),
    diff2: [stages.len]f64 = @splat(0),
    ref2: [stages.len]f64 = @splat(0),
    max_abs: [stages.len]f64 = @splat(0),
    overrun: bool = false,

    fn record(context: *anyopaque, which: fc.Stage, rows: []const f32) void {
        const self: *Comparer = @ptrCast(@alignCast(context));
        const i = @backingInt(which);
        const want = self.want.rows[i].items;
        if (self.at[i] + rows.len > want.len) {
            self.overrun = true;
            return;
        }
        for (rows, want[self.at[i]..][0..rows.len]) |g, w| {
            const d = @as(f64, g) - w;
            self.diff2[i] += d * d;
            self.ref2[i] += @as(f64, w) * w;
            self.max_abs[i] = @max(self.max_abs[i], @abs(d));
        }
        self.at[i] += rows.len;
    }
    fn relative(self: *const Comparer, i: usize) f64 {
        return @sqrt(self.diff2[i] / @max(self.ref2[i], 1e-30));
    }
    fn complete(self: *const Comparer) bool {
        if (self.overrun) return false;
        for (self.at, self.want.rows) |at, rows| if (at != rows.items.len) return false;
        return true;
    }
};

/// Pushes `frames` in pieces of `size` (0: all at once) from a reset stream.
fn decode(stream: *fm.Stream, frames: []const fc.Frame, size: usize, out: []f32, observer: ?fc.Observer) !void {
    stream.reset();
    const step = if (size == 0) frames.len else size;
    var at: usize = 0;
    while (at < frames.len) {
        const n = @min(step, frames.len - at);
        try stream.pushObserved(frames[at..][0..n], out[at * fc.samples_per_frame ..][0 .. n * fc.samples_per_frame], observer);
        at += n;
    }
}

/// Runs both cases; true when every stage of every push size is within
/// `bound` of the CPU's and the push sizes agree bit for bit.
pub fn run(gpa: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, mapped: *const inference.weights.Mapped, fixtures: []const u8, bound: f64) !bool {
    const index_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ fixtures, "index.json" }), arena, .limited(4 << 20));
    const index = try std.json.parseFromSliceLeaky(std.json.Value, arena, index_text, .{});
    const cases = index.object.get("cases").?.object;
    var rt = try fc.Runtime.init(gpa, io, mapped.document, mapped.view());
    defer rt.deinit();
    var message: [512]u8 = @splat(0);
    var gpu = inference.metal.Backend.init(gpa, &message) catch |err| {
        std.debug.print("Metal: {s} {s}\n", .{ @errorName(err), std.mem.sliceTo(&message, 0) });
        return err;
    };
    defer gpu.deinit();
    const load_start = std.Io.Clock.awake.now(io);
    var decoder = try fm.Decoder.init(gpa, &gpu, &rt);
    defer decoder.deinit();
    var stream = try fm.Stream.init(&decoder);
    defer stream.deinit();
    std.debug.print("Metal decoder: {d} MB of buffers, copied in {d:.0} ms\n", .{ decoder.bytes / 1_000_000, elapsedMs(io, load_start) });

    var ok = true;
    for ([_][]const u8{ "codec-reference", "codec-window" }) |name| {
        const files = cases.get(name).?.object.get("files").?.object;
        const codes = std.mem.bytesAsSlice(i32, try read(arena, io, fixtures, files.get("codes").?));
        const frames = try arena.alloc(fc.Frame, codes.len / fc.codebooks);
        for (frames, 0..) |*frame, t| for (frame, 0..) |*c, k| {
            c.* = @intCast(codes[t * fc.codebooks + k]);
        };
        const count = frames.len * fc.samples_per_frame;

        var want: Recorder = .{ .gpa = gpa };
        defer want.deinit();
        const cpu_out = try gpa.alloc(f32, count);
        defer gpa.free(cpu_out);
        const cpu_start = std.Io.Clock.awake.now(io);
        try rt.decodeObserved(gpa, frames, cpu_out, .{ .context = &want, .record = Recorder.record });
        const cpu_ms = elapsedMs(io, cpu_start);

        var compared: [pushes.len]Comparer = undefined;
        var outs: [pushes.len][]f32 = undefined;
        for (&compared, &outs, pushes) |*c, *out, size| {
            c.* = .{ .want = &want };
            out.* = try arena.alloc(f32, count);
            try decode(&stream, frames, size, out.*, .{ .context = c, .record = Comparer.record });
        }
        std.debug.print("{s}: {d} frames, CPU {d:.1} ms a frame; relative RMS against the CPU, pushes of 1 / 4 / {d}:\n", .{ name, frames.len, cpu_ms / @as(f64, @floatFromInt(frames.len)), frames.len });
        for (stages, 0..) |s, i| {
            var pass = true;
            for (compared) |c| pass = pass and c.relative(i) <= bound;
            ok = ok and pass;
            std.debug.print("  {s:<12} {e:.2} / {e:.2} / {e:.2}  max |d| {e:.2}{s}\n", .{ @tagName(s), compared[0].relative(i), compared[1].relative(i), compared[2].relative(i), @max(compared[0].max_abs[i], @max(compared[1].max_abs[i], compared[2].max_abs[i])), if (pass) "" else "  FAIL" });
        }
        for (compared) |c| if (!c.complete()) {
            std.debug.print("  the Metal stream observed other rows than the CPU's  FAIL\n", .{});
            ok = false;
        };
        const same = std.mem.eql(f32, outs[0], outs[2]) and std.mem.eql(f32, outs[1], outs[2]);
        std.debug.print("  pushes of 1, 4 and the whole equal bit for bit: {s}\n", .{if (same) "yes" else "no  FAIL"});
        ok = ok and same;

        // Timed unobserved, after the observed runs warmed the pipelines.
        const timed = try arena.alloc(f32, count);
        for (pushes) |size| {
            const start = std.Io.Clock.awake.now(io);
            try decode(&stream, frames, size, timed, null);
            const ms = elapsedMs(io, start);
            const equal = std.mem.eql(f32, timed, outs[2]);
            ok = ok and equal;
            std.debug.print("  pushes of {d}: {d:.2} ms a frame ({d:.0} ms){s}\n", .{ if (size == 0) frames.len else size, ms / @as(f64, @floatFromInt(frames.len)), ms, if (equal) "" else ", samples differ from the observed run  FAIL" });
        }
    }
    return ok;
}

fn read(arena: std.mem.Allocator, io: std.Io, fixtures: []const u8, entry: std.json.Value) ![]align(4) u8 {
    const path = try std.fs.path.join(arena, &.{ fixtures, entry.object.get("file").?.string });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(64 << 20));
    const aligned = try arena.alignedAlloc(u8, .@"4", bytes.len);
    @memcpy(aligned, bytes);
    return aligned;
}

fn elapsedMs(io: std.Io, start: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds)) / 1e6;
}
