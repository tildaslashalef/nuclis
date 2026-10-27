//! Mixture of experts: routing, gathered projections, and combine: the host side of `kernels/experts.metal`. Methods of `Backend`,
//! which `root.zig` aliases so callers write `backend.route(…)`.
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
const simdgroups_per_matvec_group = @import("matvec.zig").simdgroups_per_matvec_group;
const blockAligned = @import("matmul.zig").blockAligned;
const specializedMatvec = Backend.specializedMatvec;

// ----- Mixture of experts: routing, gathered projections, combine. -----

pub const RouteParams = extern struct { experts: u32, k: u32, rows: u32, in_stride: u32 };
/// Bounds of `route`: one logit per thread of the 256-thread group, and
/// the selected-probability scratch (`NU_ROUTE_MAX_K`).
pub const route_max_experts = 256;
pub const route_max_k = 64;
/// For each of `rows` logit rows (`[row][in_stride]`), the `k` experts
/// with the largest logits by (value desc, index asc) into `indices`
/// (`[row][k]` u32) and their softmax probabilities renormalized to sum
/// one into `weights` (`[row][k]` f32): `cpu.experts.route` per row.
pub fn route(self: *Backend, logits: Buffer, experts: usize, k: usize, rows: usize, in_stride: usize, indices: Buffer, weights: Buffer) !void {
    if (experts == 0 or experts > route_max_experts or k == 0 or k > experts or k > route_max_k or rows == 0 or in_stride < experts) return error.InvalidShape;
    if (logits.len < ((rows - 1) * in_stride + experts) * 4 or indices.len < rows * k * 4 or weights.len < rows * k * 4 or indices.offset % 4 != 0 or weights.offset % 4 != 0) return error.InvalidShape;
    const p: RouteParams = .{ .experts = @intCast(experts), .k = @intCast(k), .rows = @intCast(rows), .in_stride = @intCast(in_stride) };
    try self.dispatch(.route, &.{ logits, indices, weights }, p, @intCast(rows), 256, .{});
}

pub const ExpertMatvecParams = extern struct { columns: u32, stride: u32, rows: u32, blocks: u32, encoding: u32, experts: u32, slots: u32, in_stride: u32, out_stride: u32, row_groups: u32 };
/// Gathered projection: for each slot `s < slots`,
/// `output[s·out_stride + r] = W[indices[s]][r] · input[s·in_stride]`
/// over the expert matrices of `tensor` (`cpu.ExpertMatrix`, experts
/// contiguous). `in_stride` 0 shares one input across the slots.
/// Specialized bodies apply under `matvec`'s alignment rules (the input
/// stride must keep every slot float4-aligned); otherwise the generic
/// decoder. Reads only the selected experts' bytes.
pub fn matvecExperts(self: *Backend, weights: Buffer, tensor: cpu.ExpertMatrix, indices: Buffer, slots: usize, input: Buffer, in_stride: usize, output: Buffer, out_stride: usize) !void {
    const per_expert = tensor.expertBytes() catch return error.InvalidShape;
    const stride = per_expert / tensor.rows;
    if (tensor.columns % 16 != 0 or tensor.rows > std.math.maxInt(u32) / 4 or tensor.experts > std.math.maxInt(u32)) return error.InvalidShape;
    if (slots == 0 or slots > std.math.maxInt(u16) or out_stride < tensor.rows or (in_stride != 0 and in_stride < tensor.columns) or in_stride % 4 != 0) return error.InvalidShape;
    if (weights.len < tensor.bytes.len or indices.len < slots * 4 or indices.offset % 4 != 0) return error.InvalidShape;
    if (input.len < ((slots - 1) * in_stride + tensor.columns) * 4 or output.len < ((slots - 1) * out_stride + tensor.rows) * 4 or output.offset % 4 != 0) return error.InvalidShape;
    const specialized = !self.generic_only and specializedMatvec(tensor.encoding, weights.offset, stride, input.offset) != null;
    const row_groups = @divCeil(tensor.rows, 16);
    const p: ExpertMatvecParams = .{
        .columns = @intCast(tensor.columns),
        .stride = std.math.cast(u32, stride) orelse return error.InvalidShape,
        .rows = @intCast(tensor.rows),
        .blocks = @intCast(tensor.columns / 256),
        .encoding = tensor.encoding | (if (specialized) @as(u32, 0) else 0x80000000),
        .experts = @intCast(tensor.experts),
        .slots = @intCast(slots),
        .in_stride = @intCast(in_stride),
        .out_stride = std.math.cast(u32, out_stride) orelse return error.InvalidShape,
        .row_groups = @intCast(row_groups),
    };
    // The bytes that bound the dispatch are the selected experts', read once each.
    const shape: Shape = .{ .encoding = tensor.encoding, .rows = @intCast(tensor.rows * slots), .columns = @intCast(tensor.columns), .bytes = @as(u64, per_expert) * slots };
    try self.dispatch(.matvec_experts, &.{ weights, input, output, indices }, p, @intCast(slots * row_groups), 32 * simdgroups_per_matvec_group, shape);
}

pub const CombineParams = extern struct { columns: u32, slots: u32, rows: u32, in_stride: u32, out_stride: u32, experts: u32, flags: u32 };
pub const CombineShape = struct { columns: usize, slots: usize, rows: usize = 1, experts: usize, in_stride: usize, out_stride: usize };
/// `output[row][c] = Σ_s weights[row][s] · scale[indices[row][s]] · values[row·slots + s][c]`
/// (the scale factor only when `scales` is given): the weighted sum of
/// the slots' down projections into one row per token.
pub fn combineExperts(self: *Backend, values: Buffer, weights: Buffer, indices: Buffer, scales: ?Buffer, output: Buffer, s: CombineShape) !void {
    if (s.columns == 0 or s.slots == 0 or s.rows == 0 or s.experts == 0 or s.in_stride < s.columns or s.out_stride < s.columns) return error.InvalidShape;
    const slot_rows = s.rows * s.slots;
    if (values.len < ((slot_rows - 1) * s.in_stride + s.columns) * 4 or weights.len < slot_rows * 4 or indices.len < slot_rows * 4 or indices.offset % 4 != 0) return error.InvalidShape;
    if (output.len < ((s.rows - 1) * s.out_stride + s.columns) * 4) return error.InvalidShape;
    if (scales) |sc| if (sc.len < s.experts * 4) return error.InvalidShape;
    const p: CombineParams = .{ .columns = @intCast(s.columns), .slots = @intCast(s.slots), .rows = @intCast(s.rows), .in_stride = @intCast(s.in_stride), .out_stride = @intCast(s.out_stride), .experts = @intCast(s.experts), .flags = if (scales != null) 1 else 0 };
    try self.dispatch(.combine_experts, &.{ values, weights, indices, scales orelse weights, output }, p, perElement(s.columns * s.rows), 256, .{});
}

pub const GeluRowsParams = extern struct { width: u32, rows: u32, gate_stride: u32, up_stride: u32, out_stride: u32 };
/// `output[r][i] = gelu(gate[r][i]) · up[r][i]` over `rows` strided rows
/// of `width`; with a fused gate-up row, `up` is the gate buffer sliced
/// at the up half. The output must not overlap the inputs.
pub fn geluMulRows(self: *Backend, gate: Buffer, up: Buffer, output: Buffer, width: usize, rows: usize, gate_stride: usize, up_stride: usize, out_stride: usize) !void {
    if (width == 0 or rows == 0 or gate_stride < width or up_stride < width or out_stride < width) return error.InvalidShape;
    if (gate.len < ((rows - 1) * gate_stride + width) * 4 or up.len < ((rows - 1) * up_stride + width) * 4 or output.len < ((rows - 1) * out_stride + width) * 4) return error.InvalidShape;
    const out_len = ((rows - 1) * out_stride + width) * 4;
    if (Buffer.overlaps(output, out_len, gate, ((rows - 1) * gate_stride + width) * 4) or Buffer.overlaps(output, out_len, up, ((rows - 1) * up_stride + width) * 4)) return error.InvalidShape;
    const p: GeluRowsParams = .{ .width = @intCast(width), .rows = @intCast(rows), .gate_stride = @intCast(gate_stride), .up_stride = @intCast(up_stride), .out_stride = @intCast(out_stride) };
    try self.dispatch(.gelu_mul_rows, &.{ gate, up, output }, p, perElement(width * rows), 256, .{});
}

// Prefill: the chunk's slot rows grouped by expert, then gathered tiles.

pub const ExpertListsParams = extern struct { experts: u32, n: u32, max_tiles: u32 };
/// Slot rows per gathered tile; the tile bound below counts them.
pub const expert_tile = 32;
/// Tiles the lists of `n` slot rows over `experts` can hold at most:
/// Σ ceil(count/32) ≤ n/32 + experts. This bounds the gathered matmul
/// grid before the counts are known.
pub fn expertTileBound(n: usize, experts: usize) usize {
    return @divCeil(n, expert_tile) + experts;
}
/// Words of the lists buffer, in order: the tile count, `n`, the
/// `experts + 1` exclusive prefix offsets, `expertTileBound` tiles of
/// (expert, first, count), and the `n` row list.
pub const ExpertListsLayout = struct { tiles_at: usize, rows_at: usize, words: usize };
pub fn expertListsLayout(n: usize, experts: usize) ExpertListsLayout {
    const tiles_at = experts + 3;
    const rows_at = tiles_at + 3 * expertTileBound(n, experts);
    return .{ .tiles_at = tiles_at, .rows_at = rows_at, .words = rows_at + n };
}
/// Groups the `rows · k` slot rows of `indices` (`[row][k]`, as `route`
/// writes them) by expert into `lists` (`expertListsLayout(rows · k,
/// experts).words` u32, layout above) for `matmulExperts`. One
/// threadgroup; at most 256 experts.
pub fn expertLists(self: *Backend, indices: Buffer, rows: usize, k: usize, experts: usize, lists: Buffer) !void {
    if (rows == 0 or k == 0 or experts == 0 or experts > route_max_experts) return error.InvalidShape;
    const n = std.math.mul(usize, rows, k) catch return error.InvalidShape;
    if (n > std.math.maxInt(u32) / 4) return error.InvalidShape;
    const layout = expertListsLayout(n, experts);
    if (indices.len < n * 4 or indices.offset % 4 != 0 or lists.len < layout.words * 4 or lists.offset % 4 != 0) return error.InvalidShape;
    const p: ExpertListsParams = .{ .experts = @intCast(experts), .n = @intCast(n), .max_tiles = @intCast(expertTileBound(n, experts)) };
    try self.dispatch(.expert_lists, &.{ indices, lists }, p, 1, 256, .{});
}

pub const MatmulExpertsParams = extern struct { columns: u32, encoding: u32, stride: u32, rows: u32, in_stride: u32, out_stride: u32, row_tiles: u32, experts: u32, in_group: u32, tiles_at: u32, rows_at: u32 };
/// Gathered projection over the row lists: for each slot row `s` of the
/// chunk (`rows · k` of them, expert `e_s` per the lists),
/// `output[s·out_stride + r] = W[e_s][r] · input[(s / in_group)·in_stride]`.
/// `in_group` = k shares a token's input across its slots (gate-up);
/// 1 reads one input row per slot (down). Weights are read once per
/// 32-row tile of an expert's slot rows; no padding rows are written,
/// so `output` holds exactly `rows · k` rows. Same shape rules as
/// `matmul` (columns % 64, rows % 8, float4-aligned input). Q4_0 takes
/// the half tile under the matvec alignment rules, everything else the
/// generic F32 tile. The profile attributes no bytes: the volume read
/// depends on the routing.
pub fn matmulExperts(self: *Backend, weights: Buffer, tensor: cpu.ExpertMatrix, lists: Buffer, rows: usize, k: usize, input: Buffer, in_stride: usize, in_group: usize, output: Buffer, out_stride: usize) !void {
    const per_expert = tensor.expertBytes() catch return error.InvalidShape;
    const stride = per_expert / tensor.rows;
    if (tensor.rows % 8 != 0 or tensor.columns % 64 != 0 or tensor.rows > std.math.maxInt(u32) / 4 or tensor.experts > route_max_experts) return error.InvalidShape;
    if (rows == 0 or k == 0 or in_group == 0 or in_stride < tensor.columns or out_stride < tensor.rows) return error.InvalidShape;
    const n = std.math.mul(usize, rows, k) catch return error.InvalidShape;
    if (n > std.math.maxInt(u32) / 4) return error.InvalidShape;
    const layout = expertListsLayout(n, tensor.experts);
    const input_rows = @divCeil(n, in_group);
    if (weights.len < tensor.bytes.len or lists.len < layout.words * 4 or lists.offset % 4 != 0) return error.InvalidShape;
    if (input.len < ((input_rows - 1) * in_stride + tensor.columns) * 4 or input.offset % 16 != 0 or in_stride % 4 != 0) return error.InvalidShape;
    if (output.len < ((n - 1) * out_stride + tensor.rows) * 4 or output.offset % 4 != 0) return error.InvalidShape;
    const specialized = !self.generic_only and tensor.encoding == 2 and blockAligned(tensor.encoding, weights.offset, stride);
    const kernel: Kernel = if (specialized) .matmul_experts_q4_0 else .matmul_experts;
    const tile_rows: usize = if (specialized) 64 else 32;
    const row_tiles = @divCeil(tensor.rows, tile_rows);
    const p: MatmulExpertsParams = .{
        .columns = @intCast(tensor.columns),
        .encoding = tensor.encoding,
        .stride = std.math.cast(u32, stride) orelse return error.InvalidShape,
        .rows = @intCast(tensor.rows),
        .in_stride = std.math.cast(u32, in_stride) orelse return error.InvalidShape,
        .out_stride = std.math.cast(u32, out_stride) orelse return error.InvalidShape,
        .row_tiles = @intCast(row_tiles),
        .experts = @intCast(tensor.experts),
        .in_group = std.math.cast(u32, in_group) orelse return error.InvalidShape,
        .tiles_at = @intCast(layout.tiles_at),
        .rows_at = @intCast(layout.rows_at),
    };
    const groups = std.math.cast(u32, row_tiles * expertTileBound(n, tensor.experts)) orelse return error.InvalidShape;
    try self.dispatch(kernel, &.{ weights, input, output, lists }, p, groups, 128, .{ .encoding = tensor.encoding, .rows = @intCast(tensor.rows), .columns = @intCast(tensor.columns) });
}
