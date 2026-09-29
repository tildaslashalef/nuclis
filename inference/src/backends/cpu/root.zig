//! Pure CPU numerical references, independent of model architecture and I/O.
//! These favor explicit arithmetic and bounded scratch over throughput. They
//! establish operation contracts for later kernels, not a CPU inference engine.
//! `matvec` splits its rows across `Io` tasks without changing any sum, so
//! its results are the same bits on one core or many. The exception is
//! dense.zig: F32 kernels that run an encoder at usable speed.
const std = @import("std");
const quant = @import("../../quant/decode.zig");
const vector = @import("vector.zig");
pub const rope = @import("rope.zig");
pub const dense = @import("dense.zig");
pub const attention = @import("attention.zig");
pub const recurrent = @import("recurrent.zig");
pub const experts = @import("experts.zig");
pub const hadamard = @import("hadamard.zig");
pub const ExpertMatrix = experts.ExpertMatrix;

pub const rmsNorm = vector.rmsNorm;
pub const l2Norm = vector.l2Norm;
pub const softmax = vector.softmax;
pub const sigmoid = vector.sigmoid;
pub const silu = vector.silu;
pub const gelu = vector.gelu;
pub const geluQuick = vector.geluQuick;
pub const geluErf = vector.geluErf;
pub const softplus = vector.softplus;

/// Borrowed, contiguous row-major encoded weights. No padding or transpose is
/// implied. GGUF dimension 0 corresponds to `columns`, dimension 1 to `rows`.
/// Constructing this descriptor does not validate its bytes; matvec does so.
pub const Matrix = struct {
    encoding: u32,
    rows: usize,
    columns: usize,
    bytes: []const u8,
};

pub const Error = quant.Error || vector.Error || rope.Error || error{ScratchTooSmall} || std.Io.Cancelable;

test {
    _ = vector;
    _ = rope;
    _ = dense;
    _ = attention;
    _ = recurrent;
    _ = experts;
    _ = hadamard;
}

test "L2 and text RoPE agree with pinned CPU graph fixtures" {
    const Fixture = struct {
        revision: []const u8,
        l2: []const struct { input: []const f32, epsilon: f32, output: []const f32 },
        rope_input: []const f32,
        rope_dimensions: usize,
        rope_base: f32,
        rope: []const struct { position: i32, output: []const f32 },
    };
    const alloc = std.testing.allocator;
    const fixture = try std.json.parseFromSlice(Fixture, alloc, @embedFile("fixtures/vector.json"), .{ .ignore_unknown_fields = true });
    defer fixture.deinit();
    const f = fixture.value;
    try std.testing.expectEqualStrings("7620399f58aebfd2196b74021f9581bcf7218cb9", f.revision);
    for (f.l2) |case| {
        const out = try alloc.alloc(f32, case.input.len);
        defer alloc.free(out);
        try l2Norm(case.input, out, case.epsilon);
        for (out, case.output) |got, want| try std.testing.expectApproxEqAbs(want, got, 1e-7);
    }
    const out = try alloc.alloc(f32, f.rope_input.len);
    defer alloc.free(out);
    for (f.rope) |case| {
        try rope.apply(f.rope_input, out, .{ .dimensions = f.rope_dimensions, .base = f.rope_base, .position = case.position });
        // llama.cpp evolves F32 angles by repeated multiplication. The F64
        // direct-power reference increasingly differs at large positions.
        const tolerance: f32 = if (case.position <= 127) 5e-6 else 1e-3;
        for (out[0..f.rope_dimensions], case.output[0..f.rope_dimensions]) |got, want|
            try std.testing.expectApproxEqAbs(want, got, tolerance);
        try std.testing.expectEqualSlices(f32, case.output[f.rope_dimensions..], out[f.rope_dimensions..]);
    }
}

/// Tasks `matvec` may use: one per core, bounded so their failures fit on the stack.
fn matvecTasks() usize {
    return @min(max_matvec_tasks, std.Thread.getCpuCount() catch 1);
}
const max_matvec_tasks = 64;
/// Below this many weights a matrix stays on the calling thread: spawning a
/// task costs more than its share of the rows.
const matvec_parallel_weights = 1 << 16;

/// The `scratch` length that lets `matvec` use every core on `columns`.
pub fn matvecScratch(columns: usize) usize {
    return matvecTasks() * columns;
}

/// Compute output[r] = sum(weights[r,c] * input[c]). Decode weights to F32,
/// widen both operands to F64 before multiplying, sum in column order, then
/// round each result to F32. This is a reference, not bit-identical GPU math.
/// Nonfinite values propagate; F32 output overflow may produce infinity.
///
/// Rows split across `io` tasks, as many as `scratch` holds decode rows of
/// `columns` values (`matvecScratch`) and the cores allow, each taking the
/// next chunk of rows until none remain (so faster cores take more); each
/// row is still one sum in column order, so the result does not depend on
/// the split. A `columns`-wide scratch runs every row on the calling thread.
///
/// All slices are borrowed for this call; only the scratch rows the tasks use
/// are modified. Output and scratch must not overlap one another, matrix
/// bytes, or input. No allocation or I/O occurs. All expected errors are
/// checked before modifying either output or scratch.
pub fn matvec(io: std.Io, matrix: Matrix, input: []const f32, output: []f32, scratch: []f32) Error!void {
    if (matrix.rows == 0 or matrix.columns == 0 or input.len != matrix.columns or output.len != matrix.rows)
        return error.InvalidShape;
    if (scratch.len < matrix.columns) return error.ScratchTooSmall;
    // Dividing exact lengths avoids rows*columns and byte-count overflow, even
    // for malformed descriptors. Every row must contain complete packed blocks.
    if (matrix.bytes.len % matrix.rows != 0) return error.InvalidByteLength;
    const row_bytes = matrix.bytes.len / matrix.rows;
    try quant.validateRow(matrix.encoding, row_bytes, matrix.columns);
    const weights = std.math.mul(usize, matrix.rows, matrix.columns) catch std.math.maxInt(usize);
    const tasks = if (weights < matvec_parallel_weights) 1 else @min(matvecTasks(), matrix.rows, scratch.len / matrix.columns);
    const rows: Rows = .{ .matrix = matrix, .row_bytes = row_bytes, .input = input, .output = output };
    if (tasks <= 1) return rows.run(0, matrix.rows, scratch[0..matrix.columns]);
    var failures = [_]?quant.Error{null} ** max_matvec_tasks;
    var next: std.atomic.Value(usize) = .init(0);
    const chunk = @max(1, matrix.rows / (tasks * chunks_per_task));
    var group: std.Io.Group = .init;
    for (0..tasks) |t| group.async(io, Rows.task, .{ rows, &next, chunk, scratch[t * matrix.columns ..][0..matrix.columns], &failures[t] });
    try group.await(io);
    for (failures[0..tasks]) |failure| if (failure) |err| return err;
}
/// Chunks per task: enough that the performance cores can take the chunks
/// the efficiency cores have not reached.
const chunks_per_task = 8;

/// One matvec's validated operands; `run` computes a range of its rows.
const Rows = struct {
    matrix: Matrix,
    row_bytes: usize,
    input: []const f32,
    output: []f32,

    fn run(self: Rows, first: usize, last: usize, decoded: []f32) quant.Error!void {
        for (first..last) |r| {
            // r*row_bytes is bounded by the validated total byte length.
            try quant.row(self.matrix.encoding, self.matrix.bytes[r * self.row_bytes ..][0..self.row_bytes], decoded);
            var sum: f64 = 0;
            for (decoded, self.input) |weight, value| sum += @as(f64, weight) * @as(f64, value);
            self.output[r] = @floatCast(sum);
        }
    }

    /// Takes chunks of rows from `next` until none remain.
    fn task(self: Rows, next: *std.atomic.Value(usize), chunk: usize, decoded: []f32, failure: *?quant.Error) void {
        while (true) {
            const first = next.fetchAdd(chunk, .monotonic);
            if (first >= self.output.len) return;
            self.run(first, @min(first + chunk, self.output.len), decoded) catch |err| {
                failure.* = err;
                return;
            };
        }
    }
};

test "rectangular F32 matrix uses contiguous rows and preserves extra scratch" {
    const values = [_]f32{ 1, 2, 3, -4, 5, -6 };
    var bytes: [24]u8 = undefined;
    for (values, 0..) |value, i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(value), .little);
    var output: [2]f32 = undefined;
    var scratch = [_]f32{99} ** 5;
    try matvec(std.testing.io, .{ .encoding = 0, .rows = 2, .columns = 3, .bytes = &bytes }, &.{ 2, -1, 0.5 }, &output, &scratch);
    try std.testing.expectEqualSlices(f32, &.{ 1.5, -16 }, &output);
    try std.testing.expectEqualSlices(f32, &.{ 99, 99 }, scratch[3..]);
    const halves = [_]u8{ 0, 0x38, 0, 0xc0 }; // 0.5, -2
    try matvec(std.testing.io, .{ .encoding = 1, .rows = 1, .columns = 2, .bytes = &halves }, &.{ -4, 3 }, output[0..1], &scratch);
    try std.testing.expectEqual(@as(f32, -8), output[0]);
}

test "F64 accumulation retains small terms between cancelling F32 products" {
    const values = [_]f32{ 0x1p24, 1, -0x1p24 };
    var bytes: [12]u8 = undefined;
    for (values, 0..) |value, i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(value), .little);
    var output: [1]f32 = undefined;
    var scratch: [3]f32 = undefined;
    try matvec(std.testing.io, .{ .encoding = 0, .rows = 1, .columns = 3, .bytes = &bytes }, &.{ 1, 1, 1 }, &output, &scratch);
    try std.testing.expectEqual(@as(f32, 1), output[0]);
}

test "Q8 matrix spans multiple independently scaled blocks per row" {
    var bytes = [_]u8{1} ** 136; // two rows, two blocks per row, code +1
    for ([_]u16{ 0x3800, 0x4000, 0xbc00, 0x3400 }, 0..) |scale, i|
        std.mem.writeInt(u16, bytes[i * 34 ..][0..2], scale, .little);
    var input = [_]f32{1} ** 64;
    @memset(input[32..], -2);
    var output: [2]f32 = undefined;
    var scratch: [64]f32 = undefined;
    try matvec(std.testing.io, .{ .encoding = 8, .rows = 2, .columns = 64, .bytes = &bytes }, &input, &output, &scratch);
    try std.testing.expectEqualSlices(f32, &.{ -112, -48 }, &output);
}

test "invalid matrix descriptors and buffers leave writable slices untouched" {
    var output = [_]f32{123} ** 2;
    var scratch = [_]f32{456} ** 256;
    const bytes = [_]u8{0} ** 68;
    const input = [_]f32{0} ** 256;
    const valid: Matrix = .{ .encoding = 8, .rows = 2, .columns = 32, .bytes = &bytes };
    var bad = valid;
    bad.rows = 0;
    try std.testing.expectError(error.InvalidShape, matvec(std.testing.io, bad, input[0..32], &output, &scratch));
    bad = valid;
    bad.columns = 0;
    try std.testing.expectError(error.InvalidShape, matvec(std.testing.io, bad, &.{}, &output, &scratch));
    try std.testing.expectError(error.InvalidShape, matvec(std.testing.io, valid, input[0..31], &output, &scratch));
    try std.testing.expectError(error.InvalidShape, matvec(std.testing.io, valid, input[0..32], output[0..1], &scratch));
    try std.testing.expectError(error.ScratchTooSmall, matvec(std.testing.io, valid, input[0..32], &output, scratch[0..31]));
    bad = valid;
    bad.bytes = bytes[0..67]; // unequal byte counts per row
    try std.testing.expectError(error.InvalidByteLength, matvec(std.testing.io, bad, input[0..32], &output, &scratch));
    bad.bytes = bytes[0..66]; // equal rows, but truncated blocks
    try std.testing.expectError(error.InvalidByteLength, matvec(std.testing.io, bad, input[0..32], &output, &scratch));
    bad = valid;
    bad.columns = 16; // whole tensor fits one block, but neither row does
    bad.bytes = bytes[0..34];
    try std.testing.expectError(error.InvalidRowLength, matvec(std.testing.io, bad, input[0..16], &output, &scratch));
    bad = valid;
    bad.encoding = 999;
    try std.testing.expectError(error.UnsupportedEncoding, matvec(std.testing.io, bad, input[0..32], &output, &scratch));
    bad = valid;
    bad.rows = std.math.maxInt(usize);
    bad.columns = std.math.maxInt(usize);
    try std.testing.expectError(error.InvalidShape, matvec(std.testing.io, bad, input[0..32], &output, &scratch));
    for (output) |value| try std.testing.expectEqual(@as(f32, 123), value);
    for (scratch) |value| try std.testing.expectEqual(@as(f32, 456), value);
}

test "basis vectors select pinned reference columns for every quantized encoding" {
    const Fixture = struct {
        revision: []const u8,
        rows: []const struct { encoding: u32, bytes: []const u8, values: []const f32 },
    };
    const alloc = std.testing.allocator;
    for ([_][]const u8{
        @embedFile("../../quant/fixtures/simple.json"),
        @embedFile("../../quant/fixtures/k-affine.json"),
        @embedFile("../../quant/fixtures/k-signed.json"),
        @embedFile("../../quant/fixtures/iq.json"),
    }) |json| {
        const fixture = try std.json.parseFromSlice(Fixture, alloc, json, .{});
        defer fixture.deinit();
        for (fixture.value.rows) |sample| {
            const columns = @import("../../tensor/encoding.zig").layout(sample.encoding).?.elements_per_block;
            const rows = sample.values.len / columns;
            const output = try alloc.alloc(f32, rows);
            defer alloc.free(output);
            const input = try alloc.alloc(f32, columns);
            defer alloc.free(input);
            const scratch = try alloc.alloc(f32, columns);
            defer alloc.free(scratch);
            // First, interior, and last columns expose orientation and offsets.
            for ([_]usize{ 0, columns / 2 + 1, columns - 1 }) |column| {
                @memset(input, 0);
                input[column] = 1;
                try matvec(std.testing.io, .{ .encoding = sample.encoding, .rows = rows, .columns = columns, .bytes = sample.bytes }, input, output, scratch);
                for (output, 0..) |value, r| try std.testing.expectEqual(sample.values[r * columns + column], value);
            }
        }
    }
}

test "rows split across tasks give the same bits as one thread" {
    const alloc = std.testing.allocator;
    // Q8_0 rows of 512 columns (16 blocks of 34 bytes), past the parallel threshold.
    const rows = 300;
    const columns = 512;
    const bytes = try alloc.alloc(u8, rows * 16 * 34);
    defer alloc.free(bytes);
    var prng: std.Random.DefaultPrng = .init(19);
    const random = prng.random();
    random.bytes(bytes);
    // Finite scales: a random F16 scale can be NaN or infinity.
    for (0..rows * 16) |b| std.mem.writeInt(u16, bytes[b * 34 ..][0..2], @bitCast(@as(f16, @floatCast(random.float(f32) - 0.5))), .little);
    const input = try alloc.alloc(f32, columns);
    defer alloc.free(input);
    for (input) |*x| x.* = random.float(f32) * 2 - 1;
    const matrix: Matrix = .{ .encoding = 8, .rows = rows, .columns = columns, .bytes = bytes };
    var serial: [rows]f32 = undefined;
    var split: [rows]f32 = undefined;
    const scratch = try alloc.alloc(f32, @max(matvecScratch(columns), 4 * columns));
    defer alloc.free(scratch);
    try matvec(std.testing.io, matrix, input, &serial, scratch[0..columns]);
    try matvec(std.testing.io, matrix, input, &split, scratch);
    try std.testing.expectEqualSlices(u32, @ptrCast(&serial), @ptrCast(&split));
    // A scratch between one and every core's rows uses as many tasks as it holds.
    try matvec(std.testing.io, matrix, input, &split, scratch[0 .. 3 * columns]);
    try std.testing.expectEqualSlices(u32, @ptrCast(&serial), @ptrCast(&split));
}
