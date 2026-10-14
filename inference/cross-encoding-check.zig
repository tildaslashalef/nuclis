//! Cross-encoding check: the same model's matrices decoded from two files
//! must agree to the precision of the coarser encoding. Validates new CPU
//! decoders without an external oracle (docs/models/README.md § 4). Each
//! sampled row of every matrix the files share (same name and dimensions) is
//! compared by cosine and relative RMS; a negative control compares the
//! row with a distant row of the other file: the no-signal floor that a
//! wrong codebook or bit order falls toward. Matrices narrower than one
//! quantization block (the convolution taps) are skipped. Each `--min-mean
//! ENC=x` fails the run when that encoding's mean cosine (over tensors, of
//! each tensor's worst sampled row) is below x; thresholds sit between the
//! correct decoder and a deliberately broken one (docs/models/saluki.md).
//! `--fixture OUT FILE` instead writes real blocks of FILE's codebook
//! encodings with the CPU decoder's values, the rows metal-check holds the
//! GPU decoders to.
const std = @import("std");
const inference = @import("inference");

const usage = "usage: cross-encoding-check <new.gguf> <validated.gguf> [--rows n] [--min-mean ENC=x]...\n";

const Stats = struct {
    tensors: u32 = 0,
    min_cosine: f64 = 1,
    sum_cosine: f64 = 0,
    max_relative_rms: f64 = 0,
    /// Largest |cosine| of a row against a distant row: the no-signal floor.
    max_control: f64 = 0,
};

const Worst = struct { name: []const u8, new: u32, validated: u32, cosine: f64 };

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len == 4 and std.mem.eql(u8, args[1], "--fixture")) return writeFixture(init, args[2], args[3]);
    if (args.len < 3) {
        std.debug.print(usage, .{});
        return error.InvalidArguments;
    }
    var sample_rows: usize = 16;
    var thresholds: std.ArrayList(struct { id: u32, mean: f64 }) = .empty;
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        if (i + 1 >= args.len) return error.InvalidArguments;
        if (std.mem.eql(u8, args[i], "--rows")) {
            sample_rows = try std.fmt.parseInt(usize, args[i + 1], 10);
        } else if (std.mem.eql(u8, args[i], "--min-mean")) {
            const eq = std.mem.indexOfScalar(u8, args[i + 1], '=') orelse return error.InvalidArguments;
            const id = encodingId(args[i + 1][0..eq]) orelse return error.UnknownEncoding;
            try thresholds.append(arena, .{ .id = id, .mean = try std.fmt.parseFloat(f64, args[i + 1][eq + 1 ..]) });
        } else return error.InvalidArguments;
        i += 1;
    }
    if (sample_rows < 2) return error.InvalidArguments;

    var new = try inference.weights.Mapped.open(init.gpa, init.io, args[1]);
    defer new.deinit(init.io);
    var validated = try inference.weights.Mapped.open(init.gpa, init.io, args[2]);
    defer validated.deinit(init.io);
    const new_view = new.view();
    const validated_view = validated.view();

    var by_encoding: std.AutoArrayHashMapUnmanaged(u32, Stats) = .empty;
    var worst: std.ArrayList(Worst) = .empty;
    for (new.document.tensors) |*tensor| {
        if (tensor.dimensions.len < 2 or tensor.dimensions[0] < 32) continue;
        const other = for (validated.document.tensors) |*t| {
            if (std.mem.eql(u8, t.name, tensor.name)) break t;
        } else continue;
        if (!std.mem.eql(u64, tensor.dimensions, other.dimensions)) continue;
        const width: usize = @intCast(tensor.dimensions[0]);
        const rows: usize = @intCast(tensor.elements / tensor.dimensions[0]);
        const a = try arena.alloc(f32, width);
        const b = try arena.alloc(f32, width);
        const distant = try arena.alloc(f32, width);
        var tensor_min: f64 = 1;
        var tensor_rms: f64 = 0;
        var tensor_control: f64 = 0;
        const count = @min(sample_rows, rows);
        for (0..count) |k| {
            const r = if (count > 1) k * (rows - 1) / (count - 1) else 0;
            try new_view.row(tensor, r, a);
            try validated_view.row(other, r, b);
            try validated_view.row(other, (r + rows / 2) % rows, distant);
            const cosine = similarity(a, b);
            tensor_min = @min(tensor_min, cosine);
            tensor_rms = @max(tensor_rms, relativeRms(a, b));
            tensor_control = @max(tensor_control, @abs(similarity(a, distant)));
        }
        const entry = try by_encoding.getOrPut(arena, tensor.encoding_id);
        if (!entry.found_existing) entry.value_ptr.* = .{};
        const s = entry.value_ptr;
        s.tensors += 1;
        s.min_cosine = @min(s.min_cosine, tensor_min);
        s.sum_cosine += tensor_min;
        s.max_relative_rms = @max(s.max_relative_rms, tensor_rms);
        s.max_control = @max(s.max_control, tensor_control);
        try worst.append(arena, .{ .name = tensor.name, .new = tensor.encoding_id, .validated = other.encoding_id, .cosine = tensor_min });
    }

    std.debug.print("{s:<9} {s:>7} {s:>10} {s:>10} {s:>11} {s:>9}\n", .{ "encoding", "tensors", "min cos", "mean cos", "max relRMS", "control" });
    for (by_encoding.keys(), by_encoding.values()) |id, s| {
        std.debug.print("{s:<9} {d:>7} {d:>10.5} {d:>10.5} {d:>11.4} {d:>9.4}\n", .{
            name(id), s.tensors, s.min_cosine, s.sum_cosine / @as(f64, @floatFromInt(s.tensors)), s.max_relative_rms, s.max_control,
        });
    }
    std.mem.sort(Worst, worst.items, {}, struct {
        fn less(_: void, x: Worst, y: Worst) bool {
            return x.cosine < y.cosine;
        }
    }.less);
    std.debug.print("\nlowest (min cosine over {d} sampled rows):\n", .{sample_rows});
    for (worst.items[0..@min(8, worst.items.len)]) |w|
        std.debug.print("  {s:<32} {s:>8} vs {s:<6} {d:.5}\n", .{ w.name, name(w.new), name(w.validated), w.cosine });
    var failures: u32 = 0;
    for (thresholds.items) |t| {
        const s = by_encoding.get(t.id) orelse {
            std.debug.print("{s}: no shared matrix in this encoding\n", .{name(t.id)});
            failures += 1;
            continue;
        };
        const mean = s.sum_cosine / @as(f64, @floatFromInt(s.tensors));
        if (mean < t.mean) {
            std.debug.print("{s}: mean cosine {d:.5} below {d}\n", .{ name(t.id), mean, t.mean });
            failures += 1;
        }
    }
    if (failures != 0) return error.CrossEncodingMismatch;
    std.debug.print("\ncross-encoding check passed ({d} thresholds)\n", .{thresholds.items.len});
}

fn encodingId(text: []const u8) ?u32 {
    for (0..256) |id| if (inference.encoding.layout(@intCast(id))) |l| {
        if (std.mem.eql(u8, l.name, text)) return @intCast(id);
    };
    return null;
}

/// The encodings whose decoders were validated by this check, not by
/// llama.cpp fixtures.
const fixture_encodings = [_]u32{ 10, 16, 17, 18, 19, 22, 29 };

/// Two samples per encoding: the first two blocks of row 0 of the first and
/// of the last matrix stored in it.
fn writeFixture(init: std.process.Init, out_path: []const u8, model_path: []const u8) !void {
    const arena = init.arena.allocator();
    var model = try inference.weights.Mapped.open(init.gpa, init.io, model_path);
    defer model.deinit(init.io);
    const view = model.view();
    const Row = struct { encoding: u32, source: []const u8, bytes: []const u8, values: []const f32 };
    var rows: std.ArrayList(Row) = .empty;
    for (fixture_encodings) |id| {
        var first: ?*const inference.gguf.Tensor = null;
        var last: ?*const inference.gguf.Tensor = null;
        for (model.document.tensors) |*t| if (t.encoding_id == id and t.dimensions.len == 2) {
            if (first == null) first = t;
            last = t;
        };
        for ([_]?*const inference.gguf.Tensor{ first, last }) |maybe| {
            const t = maybe orelse return error.EncodingNotInFile;
            const layout = inference.encoding.layout(id).?;
            const bytes = (try view.bytes(t))[0 .. 2 * layout.bytes_per_block];
            const values = try arena.alloc(f32, 2 * layout.elements_per_block);
            try inference.quant.row(id, bytes, values);
            try rows.append(arena, .{ .encoding = id, .source = t.name, .bytes = bytes, .values = values });
        }
    }
    const Fixture = struct { revision: []const u8, rows: []const Row };
    const fixture: Fixture = .{ .revision = "nuclis CPU decoders; blocks from ConwayResearch/Underdog-Saluki-27B-1.0 4f60eba0", .rows = rows.items };
    var buffer: std.Io.Writer.Allocating = .init(arena);
    try std.json.Stringify.value(fixture, .{}, &buffer.writer);
    try buffer.writer.writeByte('\n');
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = buffer.written() });
    std.debug.print("wrote {d} rows to {s}\n", .{ rows.items.len, out_path });
}

fn name(id: u32) []const u8 {
    return if (inference.encoding.layout(id)) |l| l.name else "?";
}

fn similarity(a: []const f32, b: []const f32) f64 {
    var dot: f64 = 0;
    var aa: f64 = 0;
    var bb: f64 = 0;
    for (a, b) |x, y| {
        dot += @as(f64, x) * y;
        aa += @as(f64, x) * x;
        bb += @as(f64, y) * y;
    }
    if (aa == 0 or bb == 0) return if (aa == bb) 1 else 0;
    return dot / @sqrt(aa * bb);
}

fn relativeRms(a: []const f32, b: []const f32) f64 {
    var diff: f64 = 0;
    var bb: f64 = 0;
    for (a, b) |x, y| {
        diff += (@as(f64, x) - y) * (@as(f64, x) - y);
        bb += @as(f64, y) * y;
    }
    return if (bb == 0) 0 else @sqrt(diff / bb);
}
