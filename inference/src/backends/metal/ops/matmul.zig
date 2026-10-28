//! Batched matrix products for prefill: the tiles and their geometry: the host side of `kernels/matmul.metal`. Methods of `Backend`,
//! which `root.zig` aliases so callers write `backend.matmulPadded(…)`.
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
const specializedMatvecRows = Backend.specializedMatvecRows;
const usesMatvecRows = Backend.usesMatvecRows;
const matvec_rows_max = Backend.matvec_rows_max;

pub const MatmulParams = extern struct { columns: u32, encoding: u32, stride: u32, rows: u32, tokens: u32, in_stride: u32, out_stride: u32, row_tiles: u32 };
/// Token padding of `matmul`: activation buffers hold a multiple of this
/// many token rows (the largest token tile any instantiation uses).
pub const matmul_tile = 64;
/// Chunk length up to which the split-K 16×8 tile serves. Set to the
/// measured crossover between the small tile and the 32×32 one.
pub const small_chunk_tokens = 24;
/// Rows up to which a 2–8-token batch with no specialized tile
/// (`specializedTile`) runs the generic multi-row matvec instead of the
/// generic tile (Qwen's 48-row β/α and 1,024-row projections).
pub const small_matrix_rows = 1024;
pub fn matmulPadded(tokens: usize) usize {
    return @divCeil(tokens, matmul_tile) * matmul_tile;
}
/// Output tile of a matmul kernel and whether its operands are rounded to
/// half; must match the instantiations in kernels/matmul.metal. The specialized
/// tiles hold half operands and accumulate in F32: 64×64 tiles, 32×32 for
/// chunks of at most 32 tokens, and a 16×8 split-K tile for chunks of at
/// most `small_chunk_tokens`, where the grid's row parallelism matters more
/// than the tile's K reuse; the generic tile keeps F32 operands in 32×32.
pub const MatmulGeometry = struct { rows: usize, tokens: usize, half: bool };
pub fn matmulGeometry(kernel: Kernel) MatmulGeometry {
    return switch (kernel) {
        .matmul => .{ .rows = 32, .tokens = 32, .half = false },
        .matmul_q3_k, .matmul_q4_k, .matmul_q5_k, .matmul_q6_k, .matmul_iq3_s, .matmul_iq4_xs, .matmul_q4_0, .matmul_q5_0, .matmul_q5_1, .matmul_pq2_0, .matmul_ptq1_0 => .{ .rows = 64, .tokens = 64, .half = true },
        .matmul_q2_k, .matmul_iq2_xxs, .matmul_iq2_xs, .matmul_iq3_xxs, .matmul_iq1_s, .matmul_iq2_s, .matmul_iq1_m => .{ .rows = 64, .tokens = 64, .half = true },
        .matmul_q3_k_8, .matmul_q4_k_8, .matmul_q5_k_8, .matmul_q6_k_8, .matmul_iq3_s_8, .matmul_iq4_xs_8, .matmul_q4_0_8, .matmul_q5_0_8, .matmul_q5_1_8, .matmul_pq2_0_8, .matmul_ptq1_0_8 => .{ .rows = 16, .tokens = 8, .half = true },
        .matmul_q2_k_8, .matmul_iq2_xxs_8, .matmul_iq2_xs_8, .matmul_iq3_xxs_8, .matmul_iq1_s_8, .matmul_iq2_s_8, .matmul_iq1_m_8 => .{ .rows = 16, .tokens = 8, .half = true },
        .matmul_q3_k_w8, .matmul_q4_k_w8, .matmul_q5_k_w8, .matmul_q6_k_w8, .matmul_iq3_s_w8, .matmul_iq4_xs_w8, .matmul_q4_0_w8, .matmul_pq2_0_w8, .matmul_ptq1_0_w8 => .{ .rows = 32, .tokens = 8, .half = true },
        .matmul_q4_k_f2, .matmul_q4_k_f2hh, .matmul_q5_k_f2, .matmul_q6_k_f2, .matmul_iq4_xs_f2, .matmul_q3_k_f2, .matmul_iq3_s_f2, .matmul_q4_0_f2, .matmul_q5_0_f2, .matmul_q5_1_f2, .matmul_q8_0_f2, .matmul_pq2_0_f2, .matmul_ptq1_0_f2, .matmul_iq4_nl_f2 => .{ .rows = 16, .tokens = 8, .half = true },
        .matmul_q2_k_f2, .matmul_iq2_xxs_f2, .matmul_iq2_xs_f2, .matmul_iq3_xxs_f2, .matmul_iq1_s_f2, .matmul_iq2_s_f2, .matmul_iq1_m_f2 => .{ .rows = 16, .tokens = 8, .half = true },
        .matmul_q4_k_f4 => .{ .rows = 32, .tokens = 8, .half = true },
        else => .{ .rows = 32, .tokens = 32, .half = true },
    };
}
/// out[t][r] = Σ_c weights[r,c] · input[t,c] for `tokens` activation rows
/// (chunked prefill). `input` holds `matmulPadded(tokens)` rows of
/// `in_stride` floats and `output` as many rows of `out_stride`; rows past
/// `tokens` are scratch the caller must ignore. Requires columns % 64 == 0,
/// rows % 8 == 0, and a float4-aligned input (offset and stride). Weights
/// are read once per token tile. Q3_K, Q4_K, Q5_K, Q6_K, IQ3_S, IQ4_XS,
/// Q4_0, Q5_0, and Q5_1 go through their specialized half-operand tile when the weight
/// range is aligned (`specializedMatmul`; Q4_K, Q5_K, Q6_K, and IQ4_XS
/// through the register-fragment tile up to `small_chunk_tokens`,
/// `specializedMatmulFrag`), every other case through the generic F32
/// tile (`matmulGeometry`).
pub fn matmul(self: *Backend, weights: Buffer, matrix: cpu.Matrix, input: Buffer, in_stride: usize, output: Buffer, out_stride: usize, tokens: usize) !void {
    return matmulImpl(self, weights, matrix, input, in_stride, output, out_stride, tokens, .auto);
}
/// Same contract as `matmul`, but always uses a matrix tile. Keeps benchmark
/// controls independent of production small-batch routing.
pub fn matmulTile(self: *Backend, weights: Buffer, matrix: cpu.Matrix, input: Buffer, in_stride: usize, output: Buffer, out_stride: usize, tokens: usize) !void {
    return matmulImpl(self, weights, matrix, input, in_stride, output, out_stride, tokens, .tile);
}
/// Same contract, forcing the wide 32×8 split-K tile (a measured
/// candidate, against `matmulTile`'s 16×8 control). Rejected
/// when the encoding or alignment has no wide body, or past its 8-token
/// tile.
pub fn matmulTile32(self: *Backend, weights: Buffer, matrix: cpu.Matrix, input: Buffer, in_stride: usize, output: Buffer, out_stride: usize, tokens: usize) !void {
    return matmulImpl(self, weights, matrix, input, in_stride, output, out_stride, tokens, .wide);
}
/// Same contract, through `kernel`, a matmul instantiation for this
/// encoding and token count the caller has checked (a measurement aid).
pub fn matmulKernel(self: *Backend, kernel: Kernel, weights: Buffer, matrix: cpu.Matrix, input: Buffer, in_stride: usize, output: Buffer, out_stride: usize, tokens: usize) !void {
    return matmulImpl(self, weights, matrix, input, in_stride, output, out_stride, tokens, .{ .forced = kernel });
}
const MatmulPolicy = union(enum) { auto, tile, wide, forced: Kernel };
fn matmulImpl(self: *Backend, weights: Buffer, matrix: cpu.Matrix, input: Buffer, in_stride: usize, output: Buffer, out_stride: usize, tokens: usize, policy: MatmulPolicy) !void {
    if (matrix.rows == 0 or matrix.rows % 8 != 0 or matrix.columns == 0 or matrix.columns % 64 != 0 or matrix.bytes.len % matrix.rows != 0) return error.InvalidShape;
    if (tokens == 0 or in_stride < matrix.columns or out_stride < matrix.rows) return error.InvalidShape;
    const stride = matrix.bytes.len / matrix.rows;
    try quant.validateRow(matrix.encoding, stride, matrix.columns);
    const padded = matmulPadded(tokens);
    if (weights.len < matrix.bytes.len or input.len < padded * in_stride * 4 or output.len < padded * out_stride * 4) return error.InvalidShape;
    if (input.offset % 16 != 0 or in_stride % 4 != 0 or output.offset % 4 != 0) return error.InvalidShape;
    // A 2-row batch is a matvec problem, not a tile one, and only the
    // specialized bodies beat the tile: the generic `nu_matvec_rows` is
    // slower there (the sweep measures both).
    if (policy == .auto and usesMatvecRows(matrix.encoding, tokens) and !self.generic_only and
        specializedMatvecRows(matrix.encoding, tokens, weights.offset, stride, input.offset) != null)
        return self.matvecRows(weights, matrix, input, in_stride, output, out_stride, tokens);
    // A matrix of few rows on the generic tile leaves the GPU nearly
    // empty (48 rows: two threadgroups); one SIMD group per row does not.
    const specialized = specializedTile(matrix.encoding, weights.offset, stride, tokens);
    if (policy == .auto and !self.generic_only and tokens >= 2 and tokens <= matvec_rows_max and matrix.rows <= small_matrix_rows and specialized == null)
        return self.matvecRows(weights, matrix, input, in_stride, output, out_stride, tokens);
    const kernel = (if (self.generic_only) null else switch (policy) {
        .auto => specialized,
        .tile => specializedMatmul(matrix.encoding, weights.offset, stride, tokens),
        .wide => specializedMatmulWide(matrix.encoding, weights.offset, stride, tokens),
        .forced => |k| k,
    }) orelse blk: {
        if (policy == .wide) return error.InvalidShape;
        break :blk .matmul;
    };
    const geometry = matmulGeometry(kernel);
    const row_tiles = @divCeil(matrix.rows, geometry.rows);
    // The buffers hold `padded` rows, a multiple of every token tile.
    const token_tiles = @divCeil(tokens, geometry.tokens);
    const p: MatmulParams = .{ .columns = @intCast(matrix.columns), .encoding = matrix.encoding, .stride = @intCast(stride), .rows = @intCast(matrix.rows), .tokens = @intCast(tokens), .in_stride = @intCast(in_stride), .out_stride = @intCast(out_stride), .row_tiles = @intCast(row_tiles) };
    // Weights are read once per token tile, so the profile attributes that many bytes.
    const shape: Shape = .{ .encoding = matrix.encoding, .rows = @intCast(matrix.rows), .columns = @intCast(matrix.columns), .bytes = matrix.bytes.len * token_tiles };
    try self.dispatch(kernel, &.{ weights, input, output }, p, @intCast(row_tiles * token_tiles), 128, shape);
}
/// Whether the row start (buffer offset) and the row stride are multiples
/// of the vector-load width the encoding's block size allows (Q4_K/Q5_K
/// 16, IQ4_XS 8, Q3_K/Q6_K/IQ3_S/Q4_0/Q5_0/Q5_1/IQ2_XXS 2). False for encodings
/// without a specialized matvec; an encoding may have a matvec and no
/// matmul tile, so each picker below answers null for its own gaps.
pub fn blockAligned(encoding: u32, weight_offset: usize, stride: usize) bool {
    const alignment: usize = switch (encoding) {
        12, 13 => 16,
        23, 29 => 8,
        10, 143 => 4,
        2, 6, 7, 11, 14, 16, 17, 18, 19, 21, 22, 142 => 2,
        else => return false,
    };
    return weight_offset % alignment == 0 and stride % alignment == 0;
}
/// Picks the wide 32×8 split-K tile for the specialized encodings, up to
/// its 8-token tile. A measured candidate that lost to the 16×8 tile;
/// nothing routes to it in production.
pub fn specializedMatmulWide(encoding: u32, weight_offset: usize, stride: usize, tokens: usize) ?Kernel {
    if (tokens == 0 or tokens > 8 or !blockAligned(encoding, weight_offset, stride)) return null;
    return switch (encoding) {
        2 => .matmul_q4_0_w8,
        142 => .matmul_pq2_0_w8,
        143 => .matmul_ptq1_0_w8,
        11 => .matmul_q3_k_w8,
        21 => .matmul_iq3_s_w8,
        12 => .matmul_q4_k_w8,
        13 => .matmul_q5_k_w8,
        14 => .matmul_q6_k_w8,
        23 => .matmul_iq4_xs_w8,
        else => null,
    };
}
/// The specialized tile production `matmul` records for this shape: the
/// fragment tile, else `specializedMatmul`'s; `null` means the generic
/// tile or, for few rows, the generic multi-row matvec.
pub fn specializedTile(encoding: u32, weight_offset: usize, stride: usize, tokens: usize) ?Kernel {
    return specializedMatmulFrag(encoding, weight_offset, stride, tokens) orelse specializedMatmul(encoding, weight_offset, stride, tokens);
}
/// Picks the register-fragment 16×8 tile for at most `small_chunk_tokens`
/// tokens: every encoding with a specialized tile under its alignment
/// rules, and IQ4_NL (18-byte blocks, 2-byte aligned), which has no
/// other specialized kernel.
pub fn specializedMatmulFrag(encoding: u32, weight_offset: usize, stride: usize, tokens: usize) ?Kernel {
    if (tokens == 0 or tokens > small_chunk_tokens) return null;
    if (encoding == 20) return if (weight_offset % 2 == 0 and stride % 2 == 0) .matmul_iq4_nl_f2 else null;
    if (!blockAligned(encoding, weight_offset, stride)) return null;
    return switch (encoding) {
        12 => .matmul_q4_k_f2,
        13 => .matmul_q5_k_f2,
        14 => .matmul_q6_k_f2,
        23 => .matmul_iq4_xs_f2,
        11 => .matmul_q3_k_f2,
        21 => .matmul_iq3_s_f2,
        2 => .matmul_q4_0_f2,
        6 => .matmul_q5_0_f2,
        7 => .matmul_q5_1_f2,
        142 => .matmul_pq2_0_f2,
        143 => .matmul_ptq1_0_f2,
        10 => .matmul_q2_k_f2,
        16 => .matmul_iq2_xxs_f2,
        17 => .matmul_iq2_xs_f2,
        18 => .matmul_iq3_xxs_f2,
        19 => .matmul_iq1_s_f2,
        22 => .matmul_iq2_s_f2,
        29 => .matmul_iq1_m_f2,
        else => null,
    };
}
/// Writes element (r, c) of an 8×8 `simdgroup_matrix` filled through
/// `thread_elements()` as 2·lane + i into `out[8r + c]` (64 floats): the
/// lane layout the register-fragment tiles assume.
pub fn fragmentLayout(self: *Backend, out: Buffer) !void {
    if (out.len < 64 * 4) return error.InvalidShape;
    try self.dispatch(.fragment_layout, &.{out}, @as(u32, 0), 1, 32, .{ .encoding = null, .rows = 8, .columns = 8, .bytes = 0 });
}
/// Picks a specialized matmul tile under the same weight alignment
/// rules (the activation tile is staged through float4 loads, which
/// `matmul` checks), first the 16×8 split-K instantiation for chunks of at
/// most `small_chunk_tokens` tokens, then the 32×32 one for at most 32.
pub fn specializedMatmul(encoding: u32, weight_offset: usize, stride: usize, tokens: usize) ?Kernel {
    if (!blockAligned(encoding, weight_offset, stride)) return null;
    if (tokens <= small_chunk_tokens) return switch (encoding) {
        2 => .matmul_q4_0_8,
        6 => .matmul_q5_0_8,
        7 => .matmul_q5_1_8,
        142 => .matmul_pq2_0_8,
        143 => .matmul_ptq1_0_8,
        11 => .matmul_q3_k_8,
        21 => .matmul_iq3_s_8,
        12 => .matmul_q4_k_8,
        13 => .matmul_q5_k_8,
        14 => .matmul_q6_k_8,
        23 => .matmul_iq4_xs_8,
        10 => .matmul_q2_k_8,
        16 => .matmul_iq2_xxs_8,
        17 => .matmul_iq2_xs_8,
        18 => .matmul_iq3_xxs_8,
        19 => .matmul_iq1_s_8,
        22 => .matmul_iq2_s_8,
        29 => .matmul_iq1_m_8,
        else => null,
    };
    if (tokens <= 32) return switch (encoding) {
        2 => .matmul_q4_0_32,
        6 => .matmul_q5_0_32,
        7 => .matmul_q5_1_32,
        142 => .matmul_pq2_0_32,
        143 => .matmul_ptq1_0_32,
        11 => .matmul_q3_k_32,
        21 => .matmul_iq3_s_32,
        12 => .matmul_q4_k_32,
        13 => .matmul_q5_k_32,
        14 => .matmul_q6_k_32,
        23 => .matmul_iq4_xs_32,
        10 => .matmul_q2_k_32,
        16 => .matmul_iq2_xxs_32,
        17 => .matmul_iq2_xs_32,
        18 => .matmul_iq3_xxs_32,
        19 => .matmul_iq1_s_32,
        22 => .matmul_iq2_s_32,
        29 => .matmul_iq1_m_32,
        else => null,
    };
    return switch (encoding) {
        2 => .matmul_q4_0,
        6 => .matmul_q5_0,
        7 => .matmul_q5_1,
        142 => .matmul_pq2_0,
        143 => .matmul_ptq1_0,
        11 => .matmul_q3_k,
        21 => .matmul_iq3_s,
        12 => .matmul_q4_k,
        13 => .matmul_q5_k,
        14 => .matmul_q6_k,
        23 => .matmul_iq4_xs,
        10 => .matmul_q2_k,
        16 => .matmul_iq2_xxs,
        17 => .matmul_iq2_xs,
        18 => .matmul_iq3_xxs,
        19 => .matmul_iq1_s,
        22 => .matmul_iq2_s,
        29 => .matmul_iq1_m,
        else => null,
    };
}
