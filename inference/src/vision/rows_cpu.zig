//! The vision projectors' batched CPU products: `rowsTimes` multiplies `n`
//! activation rows by one weight matrix with F64 accumulation, decoding
//! each weight row once, split across `io` tasks without reordering any sum.
const std = @import("std");
const cpu = @import("../backends/cpu/root.zig");
const quant = @import("../quant/decode.zig");

/// `out[t] = W · input[t] (+ bias)` for `n` rows, decoding each weight row
/// once; F64 accumulation as `cpu.matvec`. Weight rows split across `io`
/// tasks as `cpu.matvec`'s do, in chunks each task takes in turn, decoding
/// into its own row of `scratch` (one row keeps them on the calling
/// thread); every sum keeps its order.
pub fn rowsTimes(io: std.Io, matrix: cpu.Matrix, input: []const f32, n: usize, output: []f32, bias: ?[]const f32, scratch: []f32) !void {
    const row_bytes = matrix.bytes.len / matrix.rows;
    try quant.validateRow(matrix.encoding, row_bytes, matrix.columns);
    const product: RowsTimes = .{ .matrix = matrix, .row_bytes = row_bytes, .input = input, .n = n, .output = output, .bias = bias };
    const tasks: usize = @min(max_row_tasks, matrix.rows, scratch.len / matrix.columns);
    if (tasks <= 1) return product.run(0, matrix.rows, scratch[0..matrix.columns]);
    var failures: [max_row_tasks]?quant.Error = @splat(null);
    var next: std.atomic.Value(usize) = .init(0);
    const chunk = @max(1, matrix.rows / (tasks * 8));
    var group: std.Io.Group = .init;
    for (0..tasks) |t| group.async(io, RowsTimes.task, .{ product, &next, chunk, scratch[t * matrix.columns ..][0..matrix.columns], &failures[t] });
    try group.await(io);
    for (failures[0..tasks]) |failure| if (failure) |err| return err;
}
const max_row_tasks = 64;

const RowsTimes = struct {
    matrix: cpu.Matrix,
    row_bytes: usize,
    input: []const f32,
    n: usize,
    output: []f32,
    bias: ?[]const f32,

    fn run(self: RowsTimes, first: usize, last: usize, decoded: []f32) quant.Error!void {
        const m = self.matrix;
        for (first..last) |r| {
            try quant.row(m.encoding, m.bytes[r * self.row_bytes ..][0..self.row_bytes], decoded);
            const b: f64 = if (self.bias) |v| v[r] else 0;
            for (0..self.n) |t| {
                var sum: f64 = 0;
                for (decoded, self.input[t * m.columns ..][0..m.columns]) |w, x| sum += @as(f64, w) * x;
                self.output[t * m.rows + r] = @floatCast(sum + b);
            }
        }
    }

    /// Takes chunks of rows from `next` until none remain.
    fn task(self: RowsTimes, next: *std.atomic.Value(usize), chunk: usize, decoded: []f32, failure: *?quant.Error) void {
        while (true) {
            const first = next.fetchAdd(chunk, .monotonic);
            if (first >= self.matrix.rows) return;
            self.run(first, @min(first + chunk, self.matrix.rows), decoded) catch |err| {
                failure.* = err;
                return;
            };
        }
    }
};
