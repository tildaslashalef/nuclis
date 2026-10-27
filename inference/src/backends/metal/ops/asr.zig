//! Nemotron 3.5 ASR's kernels: the host side of `kernels/asr.metal`. Methods
//! of `Backend`, which `root.zig` aliases so callers write
//! `backend.asrAttention(…)`. Each checks its buffers against the shape.
const std = @import("std");
const root = @import("../root.zig");
const Backend = root.Backend;
const Buffer = root.Buffer;
const perElement = Backend.perElement;
const Matrix = @import("../../cpu/root.zig").Matrix;

pub const AsrConvParams = extern struct { rows: u32, next: u32, first: u32, count: u32, width_in: u32, width_out: u32, channels: u32, in_channels: u32, relu: u32 };
/// What `asrConv2d` computes: outputs `next … next + rows − 1` from input
/// rows `first … count − 1` (`width_in · in_channels` floats each).
pub const AsrConv = struct { rows: usize, next: usize, first: usize, count: usize, width_in: usize, width_out: usize, channels: usize, in_channels: usize, relu: bool };
/// A subsampling convolution (`nemotron_encoder.conv2d`): 3×3, stride 2,
/// causal on both axes, bias added, ReLU when `s.relu`; `weight` is
/// `[channels][3][3]`, `output` `rows × width_out × channels`.
pub fn asrConv2d(self: *Backend, input: Buffer, weight: Buffer, bias: Buffer, output: Buffer, s: AsrConv) !void {
    if (s.rows == 0 or s.channels == 0 or s.width_in == 0 or s.width_out == 0 or (s.in_channels != 1 and s.in_channels != s.channels)) return error.InvalidShape;
    // The last output reads input rows up to 2·(next + rows − 1), those below `count`.
    const last = @min(2 * (s.next + s.rows - 1) + 1, s.count);
    if (last > s.first and input.len < (last - s.first) * s.width_in * s.in_channels * 4) return error.InvalidShape;
    // The first output reads row 2·next − 2 on, which the buffer must still hold.
    const low = (2 * s.next) -| 2;
    if (low < s.count and low < s.first) return error.InvalidShape;
    if (weight.len < s.channels * 9 * 4 or bias.len < s.channels * 4 or output.len < s.rows * s.width_out * s.channels * 4) return error.InvalidShape;
    const p: AsrConvParams = .{ .rows = @intCast(s.rows), .next = @intCast(s.next), .first = @intCast(s.first), .count = @intCast(s.count), .width_in = @intCast(s.width_in), .width_out = @intCast(s.width_out), .channels = @intCast(s.channels), .in_channels = @intCast(s.in_channels), .relu = @intFromBool(s.relu) };
    try self.dispatch(.asr_conv2d, &.{ input, weight, bias, output }, p, perElement(s.rows * s.width_out * s.channels), 256, .{});
}
pub const AsrRowsParams = extern struct { width: u32, rows: u32 };
/// `x[r][i] = max(x[r][i] + bias[i], 0)` over `rows` packed rows of `width`.
pub fn asrBiasRelu(self: *Backend, x: Buffer, bias: Buffer, width: usize, rows: usize) !void {
    if (width == 0 or rows == 0 or x.len < width * rows * 4 or bias.len < width * 4) return error.InvalidShape;
    try self.dispatch(.asr_bias_relu, &.{ x, bias }, AsrRowsParams{ .width = @intCast(width), .rows = @intCast(rows) }, perElement(width * rows), 256, .{});
}
pub const AsrShiftParams = extern struct { width: u32, keep: u32, from: u32, planes: u32, plane_stride: u32 };
/// In each of `planes` planes (`plane_stride` floats apart) of rows of
/// `width`, moves rows `from … from + keep − 1` to rows `0 … keep − 1`: a
/// stream's caches dropping the rows no later chunk reads. A no-op for
/// `from == 0` or `keep == 0`.
pub fn asrShift(self: *Backend, x: Buffer, width: usize, keep: usize, from: usize, planes: usize, plane_stride: usize) !void {
    if (keep == 0 or from == 0) return;
    if (width == 0 or planes == 0 or (planes > 1 and plane_stride < (from + keep) * width)) return error.InvalidShape;
    if (x.len < ((planes - 1) * plane_stride + (from + keep) * width) * 4) return error.InvalidShape;
    const p: AsrShiftParams = .{ .width = @intCast(width), .keep = @intCast(keep), .from = @intCast(from), .planes = @intCast(planes), .plane_stride = @intCast(plane_stride) };
    try self.dispatch(.asr_shift, &.{x}, p, perElement(planes * width), 256, .{});
}
pub const AsrAttentionParams = extern struct { rows: u32, held: u32, first: u32, chunk: u32, reach: u32, width: u32, offset: u32, scale: f32 };
/// What `asrAttention` attends: `rows` queries from row `first`, over the
/// `held` cached rows before them and their own.
pub const AsrAttention = struct {
    rows: usize,
    held: usize,
    first: usize,
    /// Rows per chunk (`lookahead + 1`) and the left chunks a query sees.
    chunk: usize,
    reach: usize,
    /// `heads · 128`.
    width: usize,
    /// The relative-key row of distance 0.
    offset: usize,
    /// The relative-key rows `relk` holds.
    positions: usize,
};
/// The chunk-masked Transformer-XL attention (`nemotron_encoder.attention`)
/// of heads of 128: `keys`/`values` hold `held + rows` rows (cache, then the
/// chunk's), `relk` `positions` rows, `bias_u`/`bias_v` one row each.
pub fn asrAttention(self: *Backend, q: Buffer, keys: Buffer, values: Buffer, relk: Buffer, bias_u: Buffer, bias_v: Buffer, output: Buffer, s: AsrAttention) !void {
    if (s.rows == 0 or s.width == 0 or s.width % 128 != 0 or s.chunk == 0 or s.held > s.first) return error.InvalidShape;
    // Every distance a query can see: −(chunk − 1) … reach · chunk + chunk − 1.
    if (s.offset + 1 < s.chunk or s.offset + (s.reach + 1) * s.chunk > s.positions) return error.InvalidShape;
    const row = s.width * 4;
    if (q.len < s.rows * row or output.len < s.rows * row or keys.len < (s.held + s.rows) * row or values.len < (s.held + s.rows) * row or relk.len < s.positions * row or bias_u.len < row or bias_v.len < row) return error.InvalidShape;
    if (q.offset % 16 != 0 or keys.offset % 16 != 0 or values.offset % 16 != 0 or relk.offset % 16 != 0 or bias_u.offset % 16 != 0 or bias_v.offset % 16 != 0 or output.offset % 16 != 0) return error.InvalidShape;
    const p: AsrAttentionParams = .{ .rows = @intCast(s.rows), .held = @intCast(s.held), .first = @intCast(s.first), .chunk = @intCast(s.chunk), .reach = @intCast(s.reach), .width = @intCast(s.width), .offset = @intCast(s.offset), .scale = 1.0 / @sqrt(128.0) };
    try self.dispatch(.asr_attention, &.{ q, keys, values, relk, bias_u, bias_v, output }, p, @intCast(s.rows * (s.width / 128)), 32, .{});
}
/// `out[r][c] = wide[r][c] · σ(wide[r][width + c])` (`nemotron_encoder.glu`).
pub fn asrGlu(self: *Backend, wide: Buffer, output: Buffer, width: usize, rows: usize) !void {
    if (width == 0 or rows == 0 or wide.len < rows * 2 * width * 4 or output.len < rows * width * 4) return error.InvalidShape;
    try self.dispatch(.asr_glu, &.{ wide, output }, AsrRowsParams{ .width = @intCast(width), .rows = @intCast(rows) }, perElement(width * rows), 256, .{});
}
pub const AsrDepthwiseParams = extern struct { width: u32, rows: u32, kernel_size: u32 };
/// The causal depthwise convolution (`nemotron_encoder.depthwise`): `rows`
/// holds `kernel − 1` earlier rows and the `count` new ones; `weight` is
/// `[width][kernel]`.
pub fn asrDepthwise(self: *Backend, rows: Buffer, weight: Buffer, output: Buffer, width: usize, count: usize, kernel: usize) !void {
    if (width == 0 or count == 0 or kernel == 0 or rows.len < (kernel - 1 + count) * width * 4 or weight.len < width * kernel * 4 or output.len < count * width * 4) return error.InvalidShape;
    if (Buffer.overlaps(output, count * width * 4, rows, (kernel - 1 + count) * width * 4)) return error.InvalidShape;
    try self.dispatch(.asr_depthwise, &.{ rows, weight, output }, AsrDepthwiseParams{ .width = @intCast(width), .rows = @intCast(count), .kernel_size = @intCast(kernel) }, perElement(width * count), 256, .{});
}
/// The joint's input rows, `out[r][c] = max(e[r][c] + g[c], 0)`
/// (`nemotron_rnnt.jointInput`).
pub fn asrJoint(self: *Backend, e: Buffer, g: Buffer, output: Buffer, width: usize, rows: usize) !void {
    if (width == 0 or rows == 0 or e.len < rows * width * 4 or g.len < width * 4 or output.len < rows * width * 4) return error.InvalidShape;
    try self.dispatch(.asr_joint, &.{ e, g, output }, AsrRowsParams{ .width = @intCast(width), .rows = @intCast(rows) }, perElement(width * rows), 256, .{});
}
pub const AsrScaleParams = extern struct { count: u32, factor: f32 };
/// `x[i] += factor · y[i]`; the factor must be finite.
pub fn asrAddScaled(self: *Backend, x: Buffer, y: Buffer, count: usize, factor: f32) !void {
    if (count == 0 or !std.math.isFinite(factor) or x.len < count * 4 or y.len < count * 4) return error.InvalidShape;
    try self.dispatch(.asr_add_scaled, &.{ x, y }, AsrScaleParams{ .count = @intCast(count), .factor = factor }, perElement(count), 256, .{});
}
pub const AsrRowsMatmulParams = extern struct { columns: u32, rows: u32, in_stride: u32, out_stride: u32, epilogue: u32, factor: f32, matrices: u32 };
/// What `asrMatmulRows` does with each product `s`.
pub const AsrEpilogue = union(enum) {
    store,
    silu,
    /// `out += factor · s`: a residual add.
    accumulate: f32,
    /// Row r of the output is `s_r · σ(s_{r + rows/2})`: the convolution
    /// module's GLU, `rows / 2` wide.
    glu,
};
/// One matrix of an `asrMatmulRows` call and where its rows go.
pub const AsrProduct = struct { weights: Buffer, output: Buffer };
/// The largest batch `asrMatmulRows` takes.
pub const asr_matmul_rows_max = 8;
/// `out_i[t][r] = Σ_c w_i[r][c] · x[t][c]` for `tokens` ≤ 8 rows over one to
/// three F32 matrices of `shape` (a chunk's q, k and v at once), each weight
/// row read once, with `epilogue` applied (`.glu` takes one matrix). Runs
/// the instantiation for the next power of two of `tokens`: the rows past
/// `tokens` (up to 8 of `input` and each output) are scratch, as `matmul`'s
/// padding is. Requires rows % 4 == 0 and columns % 4 == 0.
pub fn asrMatmulRows(self: *Backend, shape: Matrix, products: []const AsrProduct, input: Buffer, in_stride: usize, out_stride: usize, tokens: usize, epilogue: AsrEpilogue) !void {
    if (shape.encoding != 0 or shape.rows == 0 or shape.rows % 4 != 0 or shape.columns == 0 or shape.columns % 4 != 0) return error.InvalidShape;
    if (products.len == 0 or products.len > 3 or (epilogue == .glu and products.len != 1)) return error.InvalidShape;
    if (tokens == 0 or tokens > asr_matmul_rows_max or in_stride < shape.columns or in_stride % 4 != 0) return error.InvalidShape;
    const width = if (epilogue == .glu) shape.rows / 2 else shape.rows;
    if (out_stride < width) return error.InvalidShape;
    const padded = std.math.ceilPowerOfTwoAssert(usize, tokens);
    if (input.len < ((padded - 1) * in_stride + shape.columns) * 4 or input.offset % 16 != 0) return error.InvalidShape;
    var buffers: [7]Buffer = undefined;
    for (0..3) |i| {
        const product = products[@min(i, products.len - 1)];
        if (product.weights.len < shape.rows * shape.columns * 4 or product.weights.offset % 16 != 0) return error.InvalidShape;
        if (product.output.len < ((padded - 1) * out_stride + width) * 4 or product.output.offset % 4 != 0) return error.InvalidShape;
        buffers[i] = product.weights;
        buffers[4 + i] = product.output;
    }
    buffers[3] = input;
    const kernel: root.Kernel = switch (padded) {
        1 => .asr_matmul_rows_t1,
        2 => .asr_matmul_rows_t2,
        4 => .asr_matmul_rows_t4,
        else => .asr_matmul_rows_t8,
    };
    const p: AsrRowsMatmulParams = .{
        .columns = @intCast(shape.columns),
        .rows = @intCast(shape.rows),
        .in_stride = @intCast(in_stride),
        .out_stride = @intCast(out_stride),
        .epilogue = switch (epilogue) {
            .store => 0,
            .silu => 1,
            .accumulate => 2,
            .glu => 3,
        },
        .factor = switch (epilogue) {
            .accumulate => |f| f,
            else => 0,
        },
        .matrices = @intCast(products.len),
    };
    // Four weight rows a SIMD group, four SIMD groups a threadgroup.
    const groups = @divCeil(shape.rows * products.len / 4, 4);
    const profiled: @import("../profile.zig").Shape = .{ .encoding = 0, .rows = @intCast(shape.rows * products.len), .columns = @intCast(shape.columns), .bytes = shape.rows * shape.columns * 4 * products.len };
    try self.dispatch(kernel, &buffers, p, @intCast(groups), 128, profiled);
}
