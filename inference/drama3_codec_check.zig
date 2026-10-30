//! drama3-check's `codec` mode: Fish's codec cases under
//! `tests/fixtures/drama3/` (`codec-reference`, `codec-window`) replayed
//! through `fish_codec_encoder` and `fish_codec` on the CPU. The encoder
//! runs on the case's samples (the window case: the reference's twice
//! over) and must give Fish's codes exactly; the decoder runs on Fish's
//! codes. Every pinned stage the runtimes observe is held to a relative
//! RMS bound over its pinned rows.
const std = @import("std");
const inference = @import("inference");
const fc = inference.audio.fish_codec;
const fe = inference.audio.fish_codec_encoder;

const encoder_stages = [_]struct { []const u8, fe.Stage }{
    .{ "enc.in_conv", .encoder_in }, .{ "enc.blk.0", .block_0 },      .{ "enc.blk.1", .block_1 },
    .{ "enc.blk.2", .block_2 },      .{ "enc.blk.3.down", .block_3 }, .{ "enc.blk.3", .transformer },
    .{ "enc.out", .encoder_out },    .{ "q.down.0", .down_0 },        .{ "q.down.1", .down_1 },
    .{ "q.pre", .pre },
};
const decoder_stages = [_]struct { []const u8, fc.Stage }{
    .{ "q.z", .quantized },     .{ "q.post", .transformer },     .{ "q.up.0", .upsample_0 },
    .{ "q.up.1", .upsample_1 }, .{ "dec.in_conv", .decoder_in }, .{ "dec.blk.0", .block_0 },
    .{ "dec.blk.1", .block_1 }, .{ "dec.blk.2", .block_2 },      .{ "dec.blk.3", .block_3 },
    .{ "samples", .samples },
};

/// Every stage's rows, concatenated over the stream's pushes.
fn Recorder(comptime Stage: type) type {
    return struct {
        gpa: std.mem.Allocator,
        rows: [std.enums.values(Stage).len]std.ArrayList(f32) = @splat(.empty),

        fn record(context: *anyopaque, which: Stage, rows: []const f32) void {
            const self: *@This() = @ptrCast(@alignCast(context));
            self.rows[@backingInt(which)].appendSlice(self.gpa, rows) catch @panic("out of memory");
        }
    };
}

/// Runs both cases; true when every stage is within `bound` and the codes are Fish's.
pub fn run(gpa: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, mapped: *const inference.weights.Mapped, fixtures: []const u8, bound: f64) !bool {
    const index_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ fixtures, "index.json" }), arena, .limited(4 << 20));
    const index = try std.json.parseFromSliceLeaky(std.json.Value, arena, index_text, .{});
    const cases = index.object.get("cases").?.object;
    var decoder = try fc.Runtime.init(gpa, io, mapped.document, mapped.view());
    defer decoder.deinit();
    var encoder = try fe.Encoder.init(gpa, io, mapped.document, mapped.view());
    defer encoder.deinit();
    const reference_files = cases.get("codec-reference").?.object.get("files").?.object;
    const reference_samples = floats(try read(arena, io, fixtures, reference_files.get("samples").?));

    var ok = true;
    for ([_][]const u8{ "codec-reference", "codec-window" }) |name| {
        const files = cases.get(name).?.object.get("files").?.object;
        const samples = if (std.mem.eql(u8, name, "codec-window")) try std.mem.concat(arena, f32, &.{ reference_samples, reference_samples }) else reference_samples;

        var encoded: Recorder(fe.Stage) = .{ .gpa = arena };
        const start = std.Io.Clock.awake.now(io);
        const codes = try encoder.encodeObserved(gpa, samples, .{ .context = &encoded, .record = Recorder(fe.Stage).record });
        defer gpa.free(codes);
        const encode_ms = elapsedMs(io, start);
        const want = std.mem.bytesAsSlice(i32, try read(arena, io, fixtures, files.get("codes").?));
        var same: usize = 0;
        for (codes, 0..) |frame, t| for (frame, 0..) |c, k| {
            same += @intFromBool(t * fc.codebooks + k < want.len and c == want[t * fc.codebooks + k]);
        };
        const exact = codes.len * fc.codebooks == want.len and same == want.len;
        std.debug.print("{s}: {d} frames encoded in {d:.0} ms; codes {d}/{d} Fish's{s}\n", .{ name, codes.len, encode_ms, same, want.len, if (exact) "" else "  FAIL" });
        ok = ok and exact;
        inline for (encoder_stages) |s| if (files.get("encoder/" ++ s[0])) |entry| {
            ok = try stage(arena, io, fixtures, "encoder", s[0], entry, encoded.rows[@backingInt(s[1])].items, bound) and ok;
        };

        const frames = try arena.alloc(fc.Frame, want.len / fc.codebooks);
        for (frames, 0..) |*frame, t| for (frame, 0..) |*c, k| {
            c.* = @intCast(want[t * fc.codebooks + k]);
        };
        const out = try arena.alloc(f32, frames.len * fc.samples_per_frame);
        var decoded: Recorder(fc.Stage) = .{ .gpa = arena };
        const decode_start = std.Io.Clock.awake.now(io);
        try decoder.decodeObserved(gpa, frames, out, .{ .context = &decoded, .record = Recorder(fc.Stage).record });
        std.debug.print("{s}: {d} frames decoded in {d:.0} ms\n", .{ name, frames.len, elapsedMs(io, decode_start) });
        inline for (decoder_stages) |s| if (files.get("decoder/" ++ s[0])) |entry| {
            ok = try stage(arena, io, fixtures, "decoder", s[0], entry, decoded.rows[@backingInt(s[1])].items, bound) and ok;
        };
    }
    return ok;
}

/// One stage against its pinned rows (`rows`: [start, end) ranges of the
/// whole stage, each `width` values).
fn stage(arena: std.mem.Allocator, io: std.Io, fixtures: []const u8, side: []const u8, name: []const u8, entry: std.json.Value, got: []const f32, bound: f64) !bool {
    const want = floats(try read(arena, io, fixtures, entry));
    const shape = entry.object.get("shape").?.array.items;
    var width: usize = 1;
    for (shape[1..]) |d| width *= @intCast(d.integer);
    var at: usize = 0;
    var diff2: f64 = 0;
    var ref2: f64 = 0;
    for (entry.object.get("rows").?.array.items) |range| {
        const begin: usize = @intCast(range.array.items[0].integer);
        const end: usize = @intCast(range.array.items[1].integer);
        if (end * width > got.len) {
            std.debug.print("  {s} {s:<12} observed {d} rows, the trace pins up to {d}  FAIL\n", .{ side, name, got.len / width, end });
            return false;
        }
        for (got[begin * width .. end * width]) |g| {
            const w = want[at];
            at += 1;
            diff2 += (@as(f64, g) - w) * (@as(f64, g) - w);
            ref2 += @as(f64, w) * w;
        }
    }
    const relative = @sqrt(diff2 / @max(ref2, 1e-30));
    const pass = at == want.len and relative <= bound;
    std.debug.print("  {s} {s:<12} relative RMS {e:.2}{s}\n", .{ side, name, relative, if (pass) "" else "  FAIL" });
    return pass;
}

fn read(arena: std.mem.Allocator, io: std.Io, fixtures: []const u8, entry: std.json.Value) ![]align(4) u8 {
    const path = try std.fs.path.join(arena, &.{ fixtures, entry.object.get("file").?.string });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(64 << 20));
    const aligned = try arena.alignedAlloc(u8, .@"4", bytes.len);
    @memcpy(aligned, bytes);
    return aligned;
}

fn floats(bytes: []align(4) const u8) []const f32 {
    return std.mem.bytesAsSlice(f32, bytes);
}

fn elapsedMs(io: std.Io, start: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds)) / 1e6;
}
