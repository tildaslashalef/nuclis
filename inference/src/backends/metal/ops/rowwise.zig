//! Row-wise operations: embedding lookup, norms, RoPE, activations, the Hadamard transform: the host side of `kernels/rowwise.metal`. Methods of `Backend`,
//! which `root.zig` aliases so callers write `backend.embed(…)`.
const std = @import("std");
const root = @import("../root.zig");
const cpu = @import("../../cpu/root.zig");
const quant = @import("../../../quant/decode.zig");
const Backend = root.Backend;
const Buffer = root.Buffer;
const Kernel = root.Kernel;
const Precision = root.Precision;
const enabled = root.enabled;
const Shape = @import("../profile.zig").Shape;
const perElement = Backend.perElement;
const GeluRowsParams = Backend.GeluRowsParams;

pub const EmbedParams = extern struct { columns: u32, encoding: u32, stride: u32, token: u32 };
pub fn embed(self: *Backend, weights: Buffer, matrix: cpu.Matrix, token: u32, output: Buffer) !void {
    if (matrix.rows == 0 or matrix.columns == 0 or matrix.columns % 16 != 0 or matrix.bytes.len % matrix.rows != 0 or token >= matrix.rows) return error.InvalidShape;
    const stride = matrix.bytes.len / matrix.rows;
    try quant.validateRow(matrix.encoding, stride, matrix.columns);
    if (weights.len < matrix.bytes.len or output.len < matrix.columns * 4) return error.InvalidShape;
    const p: EmbedParams = .{ .columns = @intCast(matrix.columns), .encoding = matrix.encoding, .stride = @intCast(stride), .token = token };
    // One row is read, so the profiled shape is 1 x columns and `stride` bytes.
    try self.dispatch(.embed, &.{ weights, output }, p, perElement(matrix.columns / 16), 256, .{ .encoding = matrix.encoding, .rows = 1, .columns = @intCast(matrix.columns), .bytes = stride });
}
pub const NormParams = extern struct { width: u32, in_stride: u32, out_stride: u32, mult_stride: u32, eps: f32, flags: u32 };
pub const Norm = struct {
    rows: usize,
    width: usize,
    in_stride: usize,
    out_stride: usize,
    eps: f32 = 1e-6,
    /// Optional per-row multiplier applied as silu(multiplier) after the norm.
    silu_multiplier: ?struct { buffer: Buffer, stride: usize } = null,
};
/// Weighted RMSNorm over `rows` rows of `width`; input and output may alias exactly.
pub fn rmsNorm(self: *Backend, input: Buffer, weight: Buffer, output: Buffer, spec: Norm) !void {
    if (spec.rows == 0 or spec.width == 0 or spec.in_stride < spec.width or spec.out_stride < spec.width or !(spec.eps > 0)) return error.InvalidShape;
    if (input.len < ((spec.rows - 1) * spec.in_stride + spec.width) * 4 or output.len < ((spec.rows - 1) * spec.out_stride + spec.width) * 4 or weight.len < spec.width * 4) return error.InvalidShape;
    var mult = input;
    var mult_stride: usize = 0;
    var flags: u32 = 0;
    if (spec.silu_multiplier) |m| {
        if (m.stride < spec.width or m.buffer.len < ((spec.rows - 1) * m.stride + spec.width) * 4) return error.InvalidShape;
        mult = m.buffer;
        mult_stride = m.stride;
        flags = 1;
    }
    const p: NormParams = .{ .width = @intCast(spec.width), .in_stride = @intCast(spec.in_stride), .out_stride = @intCast(spec.out_stride), .mult_stride = @intCast(mult_stride), .eps = spec.eps, .flags = flags };
    try self.dispatch(.rmsnorm, &.{ input, weight, output, mult }, p, @intCast(spec.rows), 256, .{});
}
pub const NormAddParams = extern struct { width: u32, in_stride: u32, out_stride: u32, eps: f32, scale: f32 };
/// `(destination + norm(input)·w) · factor` over `rows` rows in one
/// dispatch, where the plans would run `rmsNorm` + `addScale` (`input`
/// and `destination` must not overlap; the fallback normalizes `input`
/// in place row by row, as the plans do).
pub fn rmsNormAdd(self: *Backend, destination: Buffer, input: Buffer, weight: Buffer, factor: f32, spec: Norm) !void {
    if (spec.rows == 0 or spec.width == 0 or spec.in_stride < spec.width or spec.out_stride < spec.width or !(spec.eps > 0) or !std.math.isFinite(factor)) return error.InvalidShape;
    if (input.len < ((spec.rows - 1) * spec.in_stride + spec.width) * 4 or destination.len < ((spec.rows - 1) * spec.out_stride + spec.width) * 4 or weight.len < spec.width * 4) return error.InvalidShape;
    if (!self.fused_norms) {
        try self.rmsNorm(input, weight, input, .{ .rows = spec.rows, .width = spec.width, .in_stride = spec.in_stride, .out_stride = spec.in_stride, .eps = spec.eps });
        for (0..spec.rows) |row| {
            const dst = destination.slice(row * spec.out_stride * 4, spec.width * 4);
            const src = input.slice(row * spec.in_stride * 4, spec.width * 4);
            try self.addScale(dst, src, spec.width, factor);
        }
        return;
    }
    const p: NormAddParams = .{ .width = @intCast(spec.width), .in_stride = @intCast(spec.in_stride), .out_stride = @intCast(spec.out_stride), .eps = spec.eps, .scale = factor };
    try self.dispatch(.rmsnorm_add, &.{ input, weight, destination }, p, @intCast(spec.rows), 256, .{});
}
pub const AddNormParams = extern struct { width: u32, in_stride: u32, out_stride: u32, eps: f32 };
/// `residual += input` and `output = norm(residual)·w` over `rows` rows
/// in one dispatch, where the plans would run `add` + `rmsNorm`. `input`
/// shares the residual's stride and must not overlap `output`.
pub fn addRmsNorm(self: *Backend, residual: Buffer, input: Buffer, weight: Buffer, output: Buffer, spec: Norm) !void {
    if (spec.rows == 0 or spec.width == 0 or spec.in_stride < spec.width or spec.out_stride < spec.width or !(spec.eps > 0)) return error.InvalidShape;
    if (input.len < ((spec.rows - 1) * spec.in_stride + spec.width) * 4 or residual.len < ((spec.rows - 1) * spec.in_stride + spec.width) * 4 or output.len < ((spec.rows - 1) * spec.out_stride + spec.width) * 4 or weight.len < spec.width * 4) return error.InvalidShape;
    if (!self.fused_norms) {
        if (spec.in_stride == spec.width) {
            try self.add(residual, input, spec.rows * spec.width);
        } else {
            for (0..spec.rows) |row| {
                const dst = residual.slice(row * spec.in_stride * 4, spec.width * 4);
                const src = input.slice(row * spec.in_stride * 4, spec.width * 4);
                try self.add(dst, src, spec.width);
            }
        }
        return self.rmsNorm(residual, weight, output, spec);
    }
    const p: AddNormParams = .{ .width = @intCast(spec.width), .in_stride = @intCast(spec.in_stride), .out_stride = @intCast(spec.out_stride), .eps = spec.eps };
    try self.dispatch(.add_rmsnorm, &.{ residual, input, weight, output }, p, @intCast(spec.rows), 256, .{});
}
pub const NormRopeParams = extern struct { width: u32, in_stride: u32, out_stride: u32, dims: u32, position: u32, pairing: u32, eps: f32 };
/// `output = norm(input)·w` over `rows` rows followed by the rotation of
/// the leading `dims` channels at `position`, in one dispatch where the
/// plans would run `rmsNorm` + `rope`. `output` may alias `input`.
pub fn rmsNormRope(self: *Backend, input: Buffer, weight: Buffer, table: Buffer, output: Buffer, spec: Norm, dims: usize, position: usize, pairing: Pairing) !void {
    if (spec.rows == 0 or spec.width == 0 or spec.in_stride < spec.width or spec.out_stride < spec.width or !(spec.eps > 0)) return error.InvalidShape;
    if (dims == 0 or dims % 2 != 0 or dims > spec.width) return error.InvalidShape;
    if (input.len < ((spec.rows - 1) * spec.in_stride + spec.width) * 4 or output.len < ((spec.rows - 1) * spec.out_stride + spec.width) * 4 or weight.len < spec.width * 4) return error.InvalidShape;
    if (table.len < (position + 1) * (dims / 2) * 8) return error.InvalidShape;
    if (!self.fused_norms) {
        try self.rmsNorm(input, weight, output, spec);
        return self.rope(output, table, spec.rows, spec.out_stride, dims, position, pairing);
    }
    const p: NormRopeParams = .{ .width = @intCast(spec.width), .in_stride = @intCast(spec.in_stride), .out_stride = @intCast(spec.out_stride), .dims = @intCast(dims), .position = @intCast(position), .pairing = @backingInt(pairing), .eps = spec.eps };
    try self.dispatch(.rmsnorm_rope, &.{ input, weight, table, output }, p, @intCast(spec.rows), 256, .{});
}
pub const HadamardParams = extern struct { width: u32, stride: u32, rows: u32, blocks: u32, inverse: u32 };
/// The block the transform is written for (`cpu.hadamard`'s contract).
pub const hadamard_block = 1024;
/// In place over `rows` rows of `width` floats at `stride`: per 1,024-block,
/// forward `x = H (signs ⊙ x)`, inverse `x = signs ⊙ (H x)`
/// (`cpu.hadamard.forward` / `.inverse`). `signs` holds `width` floats of
/// ±1; data and signs must be float4-aligned, `width` a multiple of the block.
pub fn hadamard(self: *Backend, data: Buffer, signs: Buffer, width: usize, rows: usize, stride: usize, inverse: bool) !void {
    if (rows == 0 or width == 0 or width % hadamard_block != 0 or stride < width or stride % 4 != 0) return error.InvalidShape;
    if (data.offset % 16 != 0 or signs.offset % 16 != 0 or signs.len < width * 4 or data.len < ((rows - 1) * stride + width) * 4) return error.InvalidShape;
    const blocks = width / hadamard_block;
    if (rows * blocks > std.math.maxInt(u32)) return error.InvalidShape;
    const p: HadamardParams = .{ .width = @intCast(width), .stride = @intCast(stride), .rows = @intCast(rows), .blocks = @intCast(blocks), .inverse = @intFromBool(inverse) };
    try self.dispatch(.hadamard, &.{ data, signs }, p, @intCast(rows * blocks), 256, .{});
}
pub const GatherParams = extern struct { width: u32, groups: u32, rows: u32, in_stride: u32, out_stride: u32 };
/// dst[r][g][0..width] = src[r][map[g]][0..width] over `rows` rows of
/// `groups` vectors: a fixed permutation of the vectors inside every row.
/// `map` holds `groups` u32 indices below `groups` (a bijection is the
/// caller's contract); `dst` and `src` must not overlap.
pub fn gatherRows(self: *Backend, dst: Buffer, src: Buffer, map: Buffer, width: usize, groups: usize, rows: usize, in_stride: usize, out_stride: usize) !void {
    if (width == 0 or groups == 0 or rows == 0 or in_stride < groups * width or out_stride < groups * width) return error.InvalidShape;
    const count = rows * groups * width;
    if (count > std.math.maxInt(u32) or map.len < groups * 4 or map.offset % 4 != 0) return error.InvalidShape;
    const src_len = ((rows - 1) * in_stride + groups * width) * 4;
    const dst_len = ((rows - 1) * out_stride + groups * width) * 4;
    if (src.len < src_len or dst.len < dst_len or Buffer.overlaps(dst, dst_len, src, src_len)) return error.InvalidShape;
    const p: GatherParams = .{ .width = @intCast(width), .groups = @intCast(groups), .rows = @intCast(rows), .in_stride = @intCast(in_stride), .out_stride = @intCast(out_stride) };
    try self.dispatch(.gather_rows, &.{ dst, src, map }, p, perElement(count), 256, .{});
}
pub const L2Params = extern struct { width: u32, stride: u32, eps: f32, rows: u32, heads: u32, row_stride: u32, inside: u32 = 0 };
/// Where `eps` enters an L2 norm: `clamp` divides by max(‖x‖, eps)
/// (`cpu.l2Norm`), `inside` by sqrt(‖x‖² + eps) (Fish's `l2norm`).
pub const L2Epsilon = enum(u32) { clamp, inside };
pub fn l2Norm(self: *Backend, data: Buffer, rows: usize, width: usize, stride: usize, eps: f32) !void {
    if (rows == 0 or width == 0 or stride < width or !(eps > 0) or data.len < ((rows - 1) * stride + width) * 4) return error.InvalidShape;
    const p: L2Params = .{ .width = @intCast(width), .stride = @intCast(stride), .eps = eps, .rows = @intCast(rows), .heads = @intCast(rows), .row_stride = 0 };
    try self.dispatch(.l2norm, &.{data}, p, @intCast(rows), 32, .{});
}
/// `l2Norm` over `rows` token rows of `row_stride`, each holding `heads`
/// vectors of `width` at `stride`.
pub fn l2NormRows(self: *Backend, data: Buffer, rows: usize, heads: usize, width: usize, stride: usize, row_stride: usize, eps: f32) !void {
    return l2NormRowsWith(self, data, rows, heads, width, stride, row_stride, eps, .clamp);
}
/// `l2NormRows` with the epsilon's placement chosen.
pub fn l2NormRowsWith(self: *Backend, data: Buffer, rows: usize, heads: usize, width: usize, stride: usize, row_stride: usize, eps: f32, epsilon: L2Epsilon) !void {
    if (rows == 0 or heads == 0 or width == 0 or stride < width or row_stride < (heads - 1) * stride + width or !(eps > 0)) return error.InvalidShape;
    if (data.len < ((rows - 1) * row_stride + (heads - 1) * stride + width) * 4) return error.InvalidShape;
    const p: L2Params = .{ .width = @intCast(width), .stride = @intCast(stride), .eps = eps, .rows = @intCast(rows * heads), .heads = @intCast(heads), .row_stride = @intCast(row_stride), .inside = @backingInt(epsilon) };
    try self.dispatch(.l2norm, &.{data}, p, @intCast(rows * heads), 32, .{});
}
/// Which two channels form rotation pair `i`: `cpu.rope.Pairing`.
pub const Pairing = cpu.rope.Pairing;
pub const RopeParams = extern struct { heads: u32, head_stride: u32, dims: u32, position: u32, pairing: u32 };
/// Rotates the leading `dims` channels of `heads` heads using a (cos, sin)
/// table with `dims/2` entries per position (`ropeTable`), pairing the
/// channels split-half or adjacently as `cpu.rope.apply` does.
pub fn rope(self: *Backend, data: Buffer, table: Buffer, heads: usize, head_stride: usize, dims: usize, position: usize, pairing: Pairing) !void {
    if (heads == 0 or dims == 0 or dims % 2 != 0 or head_stride < dims or data.len < ((heads - 1) * head_stride + dims) * 4) return error.InvalidShape;
    if (table.len < (position + 1) * (dims / 2) * 8) return error.InvalidShape;
    const p: RopeParams = .{ .heads = @intCast(heads), .head_stride = @intCast(head_stride), .dims = @intCast(dims), .position = @intCast(position), .pairing = @backingInt(pairing) };
    try self.dispatch(.rope, &.{ data, table }, p, perElement(heads * dims / 2), 256, .{});
}
pub const RopeRowsParams = extern struct { heads: u32, head_stride: u32, dims: u32, position: u32, rows: u32, row_stride: u32, pairing: u32 };
/// `rope` over `rows` token rows of `row_stride` floats; row t is rotated
/// for position `position + t`.
pub fn ropeRows(self: *Backend, data: Buffer, table: Buffer, heads: usize, head_stride: usize, dims: usize, position: usize, rows: usize, row_stride: usize, pairing: Pairing) !void {
    if (rows == 0 or heads == 0 or dims == 0 or dims % 2 != 0 or head_stride < dims or row_stride < (heads - 1) * head_stride + dims) return error.InvalidShape;
    if (data.len < ((rows - 1) * row_stride + (heads - 1) * head_stride + dims) * 4 or table.len < (position + rows) * (dims / 2) * 8) return error.InvalidShape;
    const p: RopeRowsParams = .{ .heads = @intCast(heads), .head_stride = @intCast(head_stride), .dims = @intCast(dims), .position = @intCast(position), .rows = @intCast(rows), .row_stride = @intCast(row_stride), .pairing = @backingInt(pairing) };
    try self.dispatch(.rope_rows, &.{ data, table }, p, perElement(rows * heads * dims / 2), 256, .{});
}
/// Fills `table` with F64-computed (cos, sin) pairs for `positions` positions
/// of the unscaled rotation with `dims` rotary channels: the angles of
/// `cpu.rope.apply` (the same for either pairing), including its optional
/// per-pair `factors` (`dims / 2` finite positive divisors; a checkpoint's
/// 1e30 entries leave a pair in place to F32 precision).
pub fn ropeTable(table: Buffer, positions: usize, dims: usize, base: f64, factors: ?[]const f32) !void {
    if (dims == 0 or dims % 2 != 0 or table.len < positions * (dims / 2) * 8) return error.InvalidShape;
    const half = dims / 2;
    if (factors) |f| {
        if (f.len != half) return error.InvalidShape;
        for (f) |x| if (!std.math.isFinite(x) or x <= 0) return error.InvalidShape;
    }
    const out = table.floats();
    for (0..positions) |position| for (0..half) |i| {
        const factor: f64 = if (factors) |f| f[i] else 1;
        const theta = @as(f64, @floatFromInt(position)) * std.math.pow(f64, base, -@as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(half))) / factor;
        out[(position * half + i) * 2] = @floatCast(@cos(theta));
        out[(position * half + i) * 2 + 1] = @floatCast(@sin(theta));
    };
}
pub const CountParams = extern struct { count: u32 };
pub fn add(self: *Backend, x: Buffer, y: Buffer, count: usize) !void {
    if (count == 0 or x.len < count * 4 or y.len < count * 4) return error.InvalidShape;
    try self.dispatch(.add, &.{ x, y }, CountParams{ .count = @intCast(count) }, perElement(count), 256, .{});
}
/// dst[0..count] = src[0..count]; the ranges must not overlap.
pub fn copy(self: *Backend, dst: Buffer, src: Buffer, count: usize) !void {
    if (count == 0 or dst.len < count * 4 or src.len < count * 4) return error.InvalidShape;
    if (dst.id == src.id and dst.offset < src.offset + count * 4 and src.offset < dst.offset + count * 4) return error.InvalidShape;
    try self.dispatch(.copy, &.{ dst, src }, CountParams{ .count = @intCast(count) }, perElement(count), 256, .{});
}
pub fn siluMul(self: *Backend, gate: Buffer, up: Buffer, count: usize) !void {
    if (count == 0 or gate.len < count * 4 or up.len < count * 4) return error.InvalidShape;
    try self.dispatch(.silu_mul, &.{ gate, up }, CountParams{ .count = @intCast(count) }, perElement(count), 256, .{});
}
pub const DflashConvParams = extern struct { rows: u32, width: u32, half_index: u32, group: u32, groups: u32, taps_stride: u32 };
/// DFlash2's two-tap dynamic convolution over `rows` rows of `width`
/// (`dflash2.convolve`): `base` is `[2][2][width]`, `taps` holds each
/// row's `taps_stride` projected taps (`[half][tap][width / group]`).
/// `input` and `output` must not alias.
pub fn dflashConv(self: *Backend, base: Buffer, taps: Buffer, input: Buffer, output: Buffer, rows: usize, width: usize, half: usize, group: usize, taps_stride: usize) !void {
    if (rows == 0 or half > 1 or group == 0 or width % group != 0) return error.InvalidShape;
    const groups = width / group;
    if (base.len < 4 * width * 4 or taps_stride < 4 * groups or taps.len < rows * taps_stride * 4 or input.len < rows * width * 4 or output.len < rows * width * 4) return error.InvalidShape;
    const p: DflashConvParams = .{ .rows = @intCast(rows), .width = @intCast(width), .half_index = @intCast(half), .group = @intCast(group), .groups = @intCast(groups), .taps_stride = @intCast(taps_stride) };
    try self.dispatch(.dflash_conv, &.{ base, taps, input, output }, p, perElement(rows * width), 256, .{});
}
pub fn silu(self: *Backend, x: Buffer, count: usize) !void {
    if (count == 0 or x.len < count * 4) return error.InvalidShape;
    try self.dispatch(.silu_inplace, &.{x}, CountParams{ .count = @intCast(count) }, perElement(count), 256, .{});
}
/// gate[i] = gelu(gate[i]) * up[i], the tanh GELU of `cpu.gelu`.
pub fn geluMul(self: *Backend, gate: Buffer, up: Buffer, count: usize) !void {
    if (count == 0 or gate.len < count * 4 or up.len < count * 4) return error.InvalidShape;
    try self.dispatch(.gelu_mul, &.{ gate, up }, CountParams{ .count = @intCast(count) }, perElement(count), 256, .{});
}
/// gate[i] = gelu_quick(gate[i]) * up[i], with gelu_quick(x) = x·σ(1.702x).
pub fn geluQuickMul(self: *Backend, gate: Buffer, up: Buffer, count: usize) !void {
    if (count == 0 or gate.len < count * 4 or up.len < count * 4) return error.InvalidShape;
    try self.dispatch(.gelu_quick_mul, &.{ gate, up }, CountParams{ .count = @intCast(count) }, perElement(count), 256, .{});
}
/// x = gelu(x), the exact (erf) form of `cpu.geluErf`, over `count` values.
pub fn geluErf(self: *Backend, x: Buffer, count: usize) !void {
    if (count == 0 or x.len < count * 4) return error.InvalidShape;
    try self.dispatch(.gelu_erf_inplace, &.{x}, CountParams{ .count = @intCast(count) }, perElement(count), 256, .{});
}
/// out[r][i] = gelu_erf(gate[r][i]) · up[r][i] over `rows` strided rows:
/// the exact-GELU gate (ModernBERT's GeGLU) where `geluMulRows` is tanh.
pub fn geluErfMulRows(self: *Backend, gate: Buffer, up: Buffer, output: Buffer, width: usize, rows: usize, gate_stride: usize, up_stride: usize, out_stride: usize) !void {
    if (width == 0 or rows == 0 or width * rows > std.math.maxInt(u32) or gate_stride < width or up_stride < width or out_stride < width) return error.InvalidShape;
    if (gate.len < ((rows - 1) * gate_stride + width) * 4 or up.len < ((rows - 1) * up_stride + width) * 4 or output.len < ((rows - 1) * out_stride + width) * 4) return error.InvalidShape;
    const out_len = ((rows - 1) * out_stride + width) * 4;
    if (Buffer.overlaps(output, out_len, gate, ((rows - 1) * gate_stride + width) * 4) or Buffer.overlaps(output, out_len, up, ((rows - 1) * up_stride + width) * 4)) return error.InvalidShape;
    const p: GeluRowsParams = .{ .width = @intCast(width), .rows = @intCast(rows), .gate_stride = @intCast(gate_stride), .up_stride = @intCast(up_stride), .out_stride = @intCast(out_stride) };
    try self.dispatch(.gelu_erf_mul_rows, &.{ gate, up, output }, p, perElement(width * rows), 256, .{});
}
pub const ScaleParams = extern struct { count: u32, factor: f32 };
/// x[i] *= factor; the factor must be finite.
pub fn scale(self: *Backend, x: Buffer, count: usize, factor: f32) !void {
    if (count == 0 or x.len < count * 4 or !std.math.isFinite(factor)) return error.InvalidShape;
    try self.dispatch(.scale, &.{x}, ScaleParams{ .count = @intCast(count), .factor = factor }, perElement(count), 256, .{});
}
/// x[i] = (x[i] + y[i]) * factor: a residual add with a per-layer output
/// scale, rounded once after the add as the CPU reference does.
pub fn addScale(self: *Backend, x: Buffer, y: Buffer, count: usize, factor: f32) !void {
    if (count == 0 or x.len < count * 4 or y.len < count * 4 or !std.math.isFinite(factor)) return error.InvalidShape;
    try self.dispatch(.add_scale, &.{ x, y }, ScaleParams{ .count = @intCast(count), .factor = factor }, perElement(count), 256, .{});
}
/// x[i] = cap * tanh(x[i] / cap), the final logit soft-cap; cap > 0.
pub fn softcap(self: *Backend, x: Buffer, count: usize, cap: f32) !void {
    if (count == 0 or x.len < count * 4 or !std.math.isFinite(cap) or cap <= 0) return error.InvalidShape;
    try self.dispatch(.softcap, &.{x}, ScaleParams{ .count = @intCast(count), .factor = cap }, perElement(count), 256, .{});
}
pub const ClampParams = extern struct { count: u32, low: f32, high: f32 };
/// x[i] = min(max(x[i], low), high); either bound may be infinite.
pub fn clamp(self: *Backend, x: Buffer, count: usize, low: f32, high: f32) !void {
    if (count == 0 or x.len < count * 4 or std.math.isNan(low) or std.math.isNan(high) or low > high) return error.InvalidShape;
    try self.dispatch(.clamp, &.{x}, ClampParams{ .count = @intCast(count), .low = low, .high = high }, perElement(count), 256, .{});
}
pub const DeltaGatesParams = extern struct { count: u32, heads: u32 };
pub fn deltaGates(self: *Backend, alpha: Buffer, beta: Buffer, a: Buffer, bias: Buffer, heads: usize) !void {
    try self.deltaGatesRows(alpha, beta, a, bias, heads, 1);
}
/// `deltaGates` over `rows` consecutive token rows of `heads` gates each.
pub fn deltaGatesRows(self: *Backend, alpha: Buffer, beta: Buffer, a: Buffer, bias: Buffer, heads: usize, rows: usize) !void {
    const count = heads * rows;
    if (heads == 0 or rows == 0 or alpha.len < count * 4 or beta.len < count * 4 or a.len < heads * 4 or bias.len < heads * 4) return error.InvalidShape;
    try self.dispatch(.delta_gates, &.{ alpha, beta, a, bias }, DeltaGatesParams{ .count = @intCast(count), .heads = @intCast(heads) }, perElement(count), 256, .{});
}
pub const GateParams = extern struct { heads: u32, width: u32, gate_stride: u32, gate_offset: u32 };
pub fn sigmoidGate(self: *Backend, out: Buffer, gates: Buffer, heads: usize, width: usize, gate_stride: usize, gate_offset: usize) !void {
    if (heads == 0 or width == 0 or out.len < heads * width * 4 or gate_stride < gate_offset + width or gates.len < ((heads - 1) * gate_stride + gate_offset + width) * 4) return error.InvalidShape;
    const p: GateParams = .{ .heads = @intCast(heads), .width = @intCast(width), .gate_stride = @intCast(gate_stride), .gate_offset = @intCast(gate_offset) };
    try self.dispatch(.sigmoid_gate, &.{ out, gates }, p, perElement(heads * width), 256, .{});
}
