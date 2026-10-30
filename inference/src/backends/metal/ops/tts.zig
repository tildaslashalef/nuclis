//! Qwen3-TTS's kernels, the host side of `kernels/tts.metal`: the codec
//! decoder's, the BF16 matrix products of the talker and predictor, and the
//! device frame's row gather and sampler. Methods of `Backend`, which
//! `root.zig` aliases so callers write `backend.ttsGather(…)`.
const std = @import("std");
const root = @import("../root.zig");
const Backend = root.Backend;
const Buffer = root.Buffer;
const Matrix = @import("../../cpu/root.zig").Matrix;
const perElement = Backend.perElement;

fn fits(buffer: Buffer, floats: usize) bool {
    return buffer.len >= floats * 4;
}

pub const TtsCodebookParams = extern struct { frames: u32, groups: u32, size: u32, dim: u32 };
/// The split quantizer's lookup over `frames` frames of `groups` codes
/// (u32, each below `size`): `first[t]` is book 0's vector at code 0,
/// `rest[t]` books 1… at their codes summed in group order; `books` is
/// `[groups][size][dim]`. The codes are not range-checked here.
pub fn ttsCodebookSum(self: *Backend, books: Buffer, codes: Buffer, first: Buffer, rest: Buffer, frames: usize, groups: usize, size: usize, dim: usize) !void {
    if (frames == 0 or groups < 2 or size == 0 or dim == 0) return error.InvalidShape;
    if (!fits(books, groups * size * dim) or !fits(codes, frames * groups) or !fits(first, frames * dim) or !fits(rest, frames * dim)) return error.InvalidShape;
    const p: TtsCodebookParams = .{ .frames = @intCast(frames), .groups = @intCast(groups), .size = @intCast(size), .dim = @intCast(dim) };
    try self.dispatch(.tts_codebook_sum, &.{ books, codes, first, rest }, p, perElement(frames * dim), 256, .{});
}

pub const TtsGatherParams = extern struct { rows: u32, width: u32, taps: u32, out_stride: u32, base: u32, step: i32 };
/// Gathers a convolution's input rows tap-major: `out[r][j·width + i] =
/// joined[r + base + j·step][i]` for `j < taps`, zero from `taps·width` to
/// `out_stride`. Every row read must lie in `joined`.
pub fn ttsGather(self: *Backend, joined: Buffer, out: Buffer, rows: usize, width: usize, taps: usize, out_stride: usize, base: usize, step: isize) !void {
    if (rows == 0 or width == 0 or taps == 0 or out_stride < taps * width) return error.InvalidShape;
    const last_tap: isize = @as(isize, @intCast(taps - 1)) * step;
    const low = @as(isize, @intCast(base)) + @min(0, last_tap);
    const high = @as(isize, @intCast(base + rows - 1)) + @max(0, last_tap);
    if (low < 0 or !fits(joined, @as(usize, @intCast(high + 1)) * width) or !fits(out, rows * out_stride)) return error.InvalidShape;
    if (joined.id == out.id) return error.InvalidShape;
    const p: TtsGatherParams = .{ .rows = @intCast(rows), .width = @intCast(width), .taps = @intCast(taps), .out_stride = @intCast(out_stride), .base = @intCast(base), .step = @intCast(step) };
    try self.dispatch(.tts_gather, &.{ joined, out }, p, perElement(rows * out_stride), 256, .{});
}

pub const TtsDepthwiseParams = extern struct { rows: u32, channels: u32, taps: u32 };
/// `out[t][c] = Σ_j weight[c][j] · joined[t + j][c] + bias[c]` over `rows`
/// rows; `joined` holds `taps − 1` history rows, then the input.
pub fn ttsDepthwise(self: *Backend, joined: Buffer, weight: Buffer, bias: Buffer, out: Buffer, rows: usize, channels: usize, taps: usize) !void {
    if (rows == 0 or channels == 0 or taps == 0) return error.InvalidShape;
    if (!fits(joined, (rows + taps - 1) * channels) or !fits(weight, channels * taps) or !fits(bias, channels) or !fits(out, rows * channels)) return error.InvalidShape;
    const p: TtsDepthwiseParams = .{ .rows = @intCast(rows), .channels = @intCast(channels), .taps = @intCast(taps) };
    try self.dispatch(.tts_depthwise, &.{ joined, weight, bias, out }, p, perElement(rows * channels), 256, .{});
}

pub const TtsSnakeParams = extern struct { count: u32, channels: u32 };
/// SnakeBeta, `y = x + inv[c] · sin²(scale[c] · x)` with `c = i mod
/// channels`; `y` may alias `x` exactly.
pub fn ttsSnake(self: *Backend, x: Buffer, scale: Buffer, inv: Buffer, y: Buffer, count: usize, channels: usize) !void {
    if (count == 0 or channels == 0 or count % channels != 0) return error.InvalidShape;
    if (!fits(x, count) or !fits(y, count) or !fits(scale, channels) or !fits(inv, channels)) return error.InvalidShape;
    try self.dispatch(.tts_snake, &.{ x, scale, inv, y }, TtsSnakeParams{ .count = @intCast(count), .channels = @intCast(channels) }, perElement(count), 256, .{});
}

pub const TtsScaledAddParams = extern struct { count: u32, channels: u32 };
/// `x[i] += scale[i mod channels] · y[i]`.
pub fn ttsScaledAdd(self: *Backend, x: Buffer, scale: Buffer, y: Buffer, count: usize, channels: usize) !void {
    if (count == 0 or channels == 0 or count % channels != 0) return error.InvalidShape;
    if (!fits(x, count) or !fits(y, count) or !fits(scale, channels)) return error.InvalidShape;
    try self.dispatch(.tts_scaled_add, &.{ x, scale, y }, TtsScaledAddParams{ .count = @intCast(count), .channels = @intCast(channels) }, perElement(count), 256, .{});
}

pub const TtsWindowParams = extern struct { rows: u32, past: u32, heads: u32, stride: u32, window: u32, scale: f32 };
/// The most keys `ttsWindowAttention` lets a query see (4 per lane).
pub const tts_window_max = 128;
/// Sliding-window attention of `rows` queries over heads of 64: query t
/// (`[rows][heads·64]`) is key row `past + t` of `keys` and `values`
/// (`[past + rows][heads·64]`, the cached rows first) and sees the
/// `window` rows ending there.
pub fn ttsWindowAttention(self: *Backend, q: Buffer, keys: Buffer, values: Buffer, out: Buffer, rows: usize, past: usize, heads: usize, window: usize, scale: f32) !void {
    if (rows == 0 or heads == 0 or window == 0 or window > tts_window_max or !std.math.isFinite(scale)) return error.InvalidShape;
    const stride = heads * 64;
    if (!fits(q, rows * stride) or !fits(out, rows * stride) or !fits(keys, (past + rows) * stride) or !fits(values, (past + rows) * stride)) return error.InvalidShape;
    const p: TtsWindowParams = .{ .rows = @intCast(rows), .past = @intCast(past), .heads = @intCast(heads), .stride = @intCast(stride), .window = @intCast(window), .scale = scale };
    try self.dispatch(.tts_window_attention, &.{ q, keys, values, out }, p, @intCast(rows * heads), 32, .{});
}

pub const TtsConvOutParams = extern struct { rows: u32, width: u32, taps: u32, squash: u32 };
/// A causal convolution to one channel, clamped to [−1, 1]: `out[r] =
/// clamp(bias[0] + Σ_j Σ_i weight[j][i] · joined[r + j][i])`; `joined`
/// holds `taps − 1` history rows of `width`, then the input.
pub fn ttsConvOut(self: *Backend, joined: Buffer, weight: Buffer, bias: Buffer, out: Buffer, rows: usize, width: usize, taps: usize) !void {
    return convOut(self, joined, weight, bias, out, rows, width, taps, 0);
}
/// `ttsConvOut` squashed by tanh instead of the clamp (Fish's codec).
pub fn ttsConvOutTanh(self: *Backend, joined: Buffer, weight: Buffer, bias: Buffer, out: Buffer, rows: usize, width: usize, taps: usize) !void {
    return convOut(self, joined, weight, bias, out, rows, width, taps, 1);
}
fn convOut(self: *Backend, joined: Buffer, weight: Buffer, bias: Buffer, out: Buffer, rows: usize, width: usize, taps: usize, squash: u32) !void {
    if (rows == 0 or width == 0 or taps == 0) return error.InvalidShape;
    if (!fits(joined, (rows + taps - 1) * width) or !fits(weight, taps * width) or !fits(bias, 1) or !fits(out, rows)) return error.InvalidShape;
    const p: TtsConvOutParams = .{ .rows = @intCast(rows), .width = @intCast(width), .taps = @intCast(taps), .squash = squash };
    try self.dispatch(.tts_conv_out, &.{ joined, weight, bias, out }, p, perElement(rows), 256, .{});
}

pub const TtsMatvecParams = extern struct { rows: u32, columns: u32, stride: u32 };
/// `output = weights · input` for a BF16 matrix (`matrix.encoding` 30)
/// with `rows_per_simdgroup` (1, 2, or 4) rows per SIMD group. The weights
/// need 2-byte alignment, the input 16 (it is read as float4), and the
/// columns a multiple of 8.
pub fn ttsMatvecBf16(self: *Backend, weights: Buffer, matrix: Matrix, input: Buffer, output: Buffer, rows_per_simdgroup: usize) !void {
    if (matrix.encoding != 30 or matrix.rows == 0 or matrix.columns == 0 or matrix.columns % 8 != 0 or matrix.bytes.len != matrix.rows * matrix.columns * 2) return error.InvalidShape;
    if (weights.len < matrix.bytes.len or weights.offset % 2 != 0 or !fits(input, matrix.columns) or input.offset % 16 != 0 or !fits(output, matrix.rows)) return error.InvalidShape;
    const kernel: root.Kernel = switch (rows_per_simdgroup) {
        1 => .tts_matvec_bf16_r1,
        2 => .tts_matvec_bf16,
        4 => .tts_matvec_bf16_r4,
        else => return error.InvalidShape,
    };
    const p: TtsMatvecParams = .{ .rows = @intCast(matrix.rows), .columns = @intCast(matrix.columns), .stride = @intCast(matrix.columns * 2) };
    const rows_per_group = 4 * rows_per_simdgroup;
    try self.dispatch(kernel, &.{ weights, input, output }, p, @intCast(@divCeil(matrix.rows, rows_per_group)), 128, .{ .encoding = 30, .rows = @intCast(matrix.rows), .columns = @intCast(matrix.columns), .bytes = matrix.bytes.len });
}

/// `matmul`'s contract for a BF16 matrix through the generic F32 tile with
/// a BF16 decode (`nu_tts_matmul_bf16`): bit for bit the generic tile's
/// result, for `tokens` rows of `input` (buffers holding
/// `matmulPadded(tokens)` rows).
pub fn ttsMatmulBf16(self: *Backend, weights: Buffer, matrix: Matrix, input: Buffer, in_stride: usize, output: Buffer, out_stride: usize, tokens: usize) !void {
    if (matrix.encoding != 30 or matrix.rows == 0 or matrix.rows % 8 != 0 or matrix.columns == 0 or matrix.columns % 64 != 0 or matrix.bytes.len != matrix.rows * matrix.columns * 2) return error.InvalidShape;
    if (tokens == 0 or in_stride < matrix.columns or out_stride < matrix.rows) return error.InvalidShape;
    const padded = Backend.matmulPadded(tokens);
    if (weights.len < matrix.bytes.len or weights.offset % 2 != 0 or input.len < padded * in_stride * 4 or output.len < padded * out_stride * 4) return error.InvalidShape;
    if (input.offset % 16 != 0 or in_stride % 4 != 0 or output.offset % 4 != 0) return error.InvalidShape;
    const row_tiles = @divCeil(matrix.rows, 32);
    const token_tiles = @divCeil(tokens, 32);
    const p: Backend.MatmulParams = .{ .columns = @intCast(matrix.columns), .encoding = 30, .stride = @intCast(matrix.columns * 2), .rows = @intCast(matrix.rows), .tokens = @intCast(tokens), .in_stride = @intCast(in_stride), .out_stride = @intCast(out_stride), .row_tiles = @intCast(row_tiles) };
    try self.dispatch(.tts_matmul_bf16, &.{ weights, input, output }, p, @intCast(row_tiles * token_tiles), 128, .{ .encoding = 30, .rows = @intCast(matrix.rows), .columns = @intCast(matrix.columns), .bytes = matrix.bytes.len * token_tiles });
}

pub const TtsEmbedRowParams = extern struct { width: u32, rows: u32, slot: u32, accumulate: u32 };
/// `out = table[codes[slot]]` (or `+=` with `accumulate`) for a BF16
/// `[rows][width]` table and u32 codes on the device.
pub fn ttsEmbedRow(self: *Backend, table: Buffer, rows: usize, width: usize, codes: Buffer, slot: usize, out: Buffer, accumulate: bool) !void {
    if (rows == 0 or width == 0 or table.len < rows * width * 2 or table.offset % 2 != 0 or !fits(codes, slot + 1) or !fits(out, width)) return error.InvalidShape;
    const p: TtsEmbedRowParams = .{ .width = @intCast(width), .rows = @intCast(rows), .slot = @intCast(slot), .accumulate = @intFromBool(accumulate) };
    try self.dispatch(.tts_embed_row, &.{ table, codes, out }, p, perElement(width), 256, .{});
}

pub const TtsSampleParams = extern struct { count: u32, slot: u32, top_k: u32, draw: u32, top_p: f32, temperature: f32 };
/// The most logits `ttsSample` sorts.
pub const tts_sample_max = 2048;
/// Samples `count` logits as the CPU sampler would (temperature, then
/// top-k, then top-p, with the uniform `draws[draw]`) in F32 and writes the
/// id to `codes[slot]` (u32). The host verifies the choice.
pub fn ttsSample(self: *Backend, logits: Buffer, count: usize, draws: Buffer, draw: usize, codes: Buffer, slot: usize, top_k: usize, top_p: f32, temperature: f32) !void {
    if (count == 0 or count > tts_sample_max or !fits(logits, count) or !fits(draws, draw + 1) or !fits(codes, slot + 1)) return error.InvalidShape;
    if (!(top_p > 0 and top_p <= 1) or !(temperature >= 0) or !std.math.isFinite(temperature)) return error.InvalidShape;
    const p: TtsSampleParams = .{ .count = @intCast(count), .slot = @intCast(slot), .top_k = @intCast(@min(top_k, count)), .draw = @intCast(draw), .top_p = top_p, .temperature = temperature };
    try self.dispatch(.tts_sample, &.{ logits, draws, codes }, p, 1, 1024, .{});
}

pub const TtsMatvec3Params = extern struct { rows0: u32, rows1: u32, rows2: u32, columns: u32 };
/// `outputs[i] = matrices[i] · input` for three BF16 matrices of the same
/// width in one dispatch (each row count a multiple of 8; the input
/// 16-byte aligned).
pub fn ttsMatvec3Bf16(self: *Backend, weights: [3]Buffer, matrices: [3]Matrix, input: Buffer, outputs: [3]Buffer) !void {
    const columns = matrices[0].columns;
    if (columns == 0 or columns % 8 != 0 or !fits(input, columns) or input.offset % 16 != 0) return error.InvalidShape;
    var bytes: u64 = 0;
    var rows: usize = 0;
    for (weights, matrices, outputs) |w, m, out| {
        if (m.encoding != 30 or m.columns != columns or m.rows == 0 or m.rows % 8 != 0 or m.bytes.len != m.rows * columns * 2) return error.InvalidShape;
        if (w.len < m.bytes.len or w.offset % 2 != 0 or !fits(out, m.rows)) return error.InvalidShape;
        bytes += m.bytes.len;
        rows += m.rows;
    }
    const p: TtsMatvec3Params = .{ .rows0 = @intCast(matrices[0].rows), .rows1 = @intCast(matrices[1].rows), .rows2 = @intCast(matrices[2].rows), .columns = @intCast(columns) };
    try self.dispatch(.tts_matvec3_bf16, &.{ weights[0], weights[1], weights[2], input, outputs[0], outputs[1], outputs[2] }, p, @intCast(rows / 8), 128, .{ .encoding = 30, .rows = @intCast(rows), .columns = @intCast(columns), .bytes = bytes });
}

pub const TtsGateUpParams = extern struct { rows: u32, columns: u32 };
/// `out = silu(gate · input) ⊙ (up · input)` for BF16 `gate` and `up` of
/// the same shape, in one dispatch.
pub fn ttsGateUpBf16(self: *Backend, gate: Buffer, up: Buffer, matrix: Matrix, input: Buffer, out: Buffer) !void {
    if (matrix.encoding != 30 or matrix.rows == 0 or matrix.columns == 0 or matrix.columns % 8 != 0 or matrix.bytes.len != matrix.rows * matrix.columns * 2) return error.InvalidShape;
    if (gate.len < matrix.bytes.len or up.len < matrix.bytes.len or gate.offset % 2 != 0 or up.offset % 2 != 0 or !fits(input, matrix.columns) or input.offset % 16 != 0 or !fits(out, matrix.rows)) return error.InvalidShape;
    const p: TtsGateUpParams = .{ .rows = @intCast(matrix.rows), .columns = @intCast(matrix.columns) };
    try self.dispatch(.tts_gate_up_bf16, &.{ gate, up, input, out }, p, @intCast(@divCeil(matrix.rows, 8)), 128, .{ .encoding = 30, .rows = @intCast(matrix.rows), .columns = @intCast(matrix.columns), .bytes = 2 * matrix.bytes.len });
}
