//! Matrix-vector products: the generic and specialized kernels, split-K, multi-row, and segments: the host side of `kernels/matvec.metal`. Methods of `Backend`,
//! which `root.zig` aliases so callers write `backend.matvec(…)`.
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
const blockAligned = @import("matmul.zig").blockAligned;

// ----- Kernel encoders. Each validates shapes before recording. -----

pub const MatvecParams = extern struct { columns: u32, encoding: u32, stride: u32, rows: u32 };
pub const MatvecBlockParams = extern struct { columns: u32, stride: u32, rows: u32, blocks: u32 };
/// Rows each SIMD group of a specialized matvec accumulates; must equal the
/// template argument of the `nu_matvec_*` instantiations in kernels/.
pub const rows_per_simdgroup = 4;
pub const simdgroups_per_matvec_group = 4;
/// output[r] = sum_c weights[r,c] * input[c]; weights are an encoded matrix
/// with `stride` bytes per row. Input/output are F32 vectors. Every
/// encoding `specializedMatvec` names uses its block kernel when the
/// weight range and input are aligned for its vector loads.
pub fn matvec(self: *Backend, weights: Buffer, matrix: cpu.Matrix, input: Buffer, output: Buffer) !void {
    return matvecImpl(self, weights, matrix, input, output, 1);
}
/// `matvec`'s contract, forcing `splits` K splits (the split sweep's knob;
/// 1 is the single-pass kernel). Only the specialized encodings with a
/// split body are served. No production shape routes here: the 2026-09-21
/// sweep measured the split path behind the single pass on every row
/// bucket, so the kernels stay as the measured fixture and the
/// single pass is what `matvec` records.
pub fn matvecSplits(self: *Backend, weights: Buffer, matrix: cpu.Matrix, input: Buffer, output: Buffer, splits: usize) !void {
    if (splits == 0 or splits > split_k_max) return error.InvalidShape;
    return matvecImpl(self, weights, matrix, input, output, splits);
}
/// Rows per SIMD group of a standalone specialized matvec: the template
/// argument of its instantiation in kernels/.
pub fn matvecRowsPerSimdgroup(kernel: Kernel) usize {
    return switch (kernel) {
        .matvec_iq2_xxs, .matvec_iq2_xs, .matvec_iq2_s, .matvec_q2_k, .matvec_iq1_m => 2,
        else => rows_per_simdgroup,
    };
}
/// SIMD groups per threadgroup of a standalone specialized matvec; must
/// match its kernel in kernels/. IQ2_S's 16 share one copy of its
/// 8 KiB grid (2.5–4 % over 4, docs/engine/apple-gpu.md).
pub fn matvecSimdgroups(kernel: Kernel) usize {
    return switch (kernel) {
        .matvec_iq2_s => 16,
        else => simdgroups_per_matvec_group,
    };
}
/// Most splits the row-partial scratch holds.
pub const split_k_max = 8;
/// Most rows the row-partial scratch holds per split (512 KB at 8 splits).
pub const split_k_rows = 16_384;
/// The split body of a specialized matvec, when one exists.
pub fn matvecSplitKernel(kernel: Kernel) ?Kernel {
    return switch (kernel) {
        .matvec_q4_k => .matvec_q4_k_split,
        .matvec_q5_k => .matvec_q5_k_split,
        else => null,
    };
}
/// Row-partial scratch, created on first use: `split_k_rows` rows per
/// split; the forced path validates the row count against it.
fn splitScratch(self: *Backend) !Buffer {
    if (self.split_scratch) |b| return b;
    const b = try self.create(split_k_rows * split_k_max * 4);
    self.split_scratch = b;
    return b;
}
fn matvecImpl(self: *Backend, weights: Buffer, matrix: cpu.Matrix, input: Buffer, output: Buffer, splits: usize) !void {
    if (matrix.rows == 0 or matrix.columns == 0 or matrix.columns % 16 != 0 or matrix.bytes.len % matrix.rows != 0) return error.InvalidShape;
    const stride = matrix.bytes.len / matrix.rows;
    try quant.validateRow(matrix.encoding, stride, matrix.columns);
    if (weights.len < matrix.bytes.len or input.len < matrix.columns * 4 or output.len < matrix.rows * 4) return error.InvalidShape;
    // Weight bytes are the traffic that bounds decode; the profile divides them by measured time.
    const shape: Shape = .{ .encoding = matrix.encoding, .rows = @intCast(matrix.rows), .columns = @intCast(matrix.columns), .bytes = matrix.bytes.len };
    const rows_per_group = rows_per_simdgroup * simdgroups_per_matvec_group;
    if (!self.generic_only) if (specializedMatvec(matrix.encoding, weights.offset, stride, input.offset)) |kernel| {
        if (splits > 1) {
            const split_kernel = matvecSplitKernel(kernel) orelse return error.InvalidShape;
            const scratch = try splitScratch(self);
            if (matrix.rows * splits * 4 > scratch.len) return error.InvalidShape;
            const blocks = matrix.columns / 256;
            const split_blocks = @divCeil(blocks, splits);
            const p: MatvecSplitParams = .{ .columns = @intCast(matrix.columns), .stride = @intCast(stride), .rows = @intCast(matrix.rows), .blocks = @intCast(blocks), .splits = @intCast(splits), .split_blocks = @intCast(split_blocks) };
            const row_groups = @divCeil(matrix.rows, rows_per_group);
            try self.dispatch(split_kernel, &.{ weights, input, scratch }, p, @intCast(row_groups * splits), 32 * simdgroups_per_matvec_group, shape);
            const rp: ReduceSplitsParams = .{ .rows = @intCast(matrix.rows), .splits = @intCast(splits) };
            try self.dispatch(.reduce_splits, &.{ scratch, output }, rp, perElement(matrix.rows), 256, .{});
            return;
        }
        const p: MatvecBlockParams = .{ .columns = @intCast(matrix.columns), .stride = @intCast(stride), .rows = @intCast(matrix.rows), .blocks = @intCast(matrix.columns / 256) };
        const simdgroups = matvecSimdgroups(kernel);
        const group_rows = matvecRowsPerSimdgroup(kernel) * simdgroups;
        try self.dispatch(kernel, &.{ weights, input, output }, p, @intCast(@divCeil(matrix.rows, group_rows)), @intCast(32 * simdgroups), shape);
        return;
    };
    if (splits > 1) return error.InvalidShape;
    const p: MatvecParams = .{ .columns = @intCast(matrix.columns), .encoding = matrix.encoding, .stride = @intCast(stride), .rows = @intCast(matrix.rows) };
    try self.dispatch(.matvec, &.{ weights, input, output }, p, @intCast(matrix.rows), 32, shape);
}
pub const MatvecSplitParams = extern struct { columns: u32, stride: u32, rows: u32, blocks: u32, splits: u32, split_blocks: u32 };
pub const ReduceSplitsParams = extern struct { rows: u32, splits: u32 };
pub const MatvecRowsParams = extern struct { columns: u32, encoding: u32, stride: u32, rows: u32, tokens: u32, in_stride: u32, out_stride: u32 };
/// Largest batch `matmul` routes to the multi-row matvec: 2 rows for every
/// encoding with a body, 3 for the K-quants. The word-outer bodies read
/// 150–214 GB/s (weight bytes) at 2 rows and Q4_K/Q5_K/Q6_K 121–158 at
/// 3, against the fragment tile's 104–151; IQ4_XS at 3 and every
/// encoding from 4 lose to the tile (`--matvec-rows-bench`, 2026-10-03).
pub const small_batch_rows = 3;
/// Whether `matmul` routes the multi-row matvec at all. Verify batches
/// of 2 and 3 rows are common: the drafter stops its chain at a
/// low-probability proposal.
pub const route_small_batch = true;
/// Largest token count the multi-row kernels are instantiated for; the
/// sweep measures every count to `matvec_rows_max` even though `matmul`
/// routes only to `small_batch_rows`.
pub const matvec_rows_max = 8;
/// Rows each SIMD group of a multi-row body accumulates: four 8-lane
/// groups of `NU_ROWS_PER_SIMDGROUP` (kernels/matvec_rows.metal) rows each.
pub const matvec_rows_per_simdgroup = 8;
pub fn usesMatvecRows(encoding: u32, tokens: usize) bool {
    const limit: usize = switch (encoding) {
        12, 13, 14 => small_batch_rows,
        else => 2,
    };
    return route_small_batch and tokens >= 2 and tokens <= limit;
}
/// output[t][r] = Σ_c weights[r,c] · input[t,c] for `1 < tokens ≤
/// matvec_rows_max` activation rows. Each weight block is decoded once
/// and multiplied against every token row, so the weight bytes (the
/// bandwidth bound of a small batch) are read once instead of once per
/// token. The specialized encodings use their multi-row body when aligned,
/// every other encoding the generic one. `input` and `output` hold
/// `tokens` rows of `in_stride`/`out_stride` floats. `matmul` routes only
/// the specialized encodings here; `metal-check` calls it directly for
/// every encoding.
pub fn matvecRows(self: *Backend, weights: Buffer, matrix: cpu.Matrix, input: Buffer, in_stride: usize, output: Buffer, out_stride: usize, tokens: usize) !void {
    if (matrix.rows == 0 or matrix.columns == 0 or matrix.columns % 16 != 0 or matrix.bytes.len % matrix.rows != 0) return error.InvalidShape;
    const stride = matrix.bytes.len / matrix.rows;
    try quant.validateRow(matrix.encoding, stride, matrix.columns);
    if (tokens < 2 or tokens > matvec_rows_max or tokens > 64) return error.InvalidShape;
    if (in_stride < matrix.columns or out_stride < matrix.rows) return error.InvalidShape;
    if (weights.len < matrix.bytes.len or input.len < ((tokens - 1) * in_stride + matrix.columns) * 4 or output.len < ((tokens - 1) * out_stride + matrix.rows) * 4) return error.InvalidShape;
    if (input.offset % 16 != 0 or in_stride % 4 != 0 or output.offset % 4 != 0) return error.InvalidShape;
    const shape: Shape = .{ .encoding = matrix.encoding, .rows = @intCast(matrix.rows), .columns = @intCast(matrix.columns), .bytes = matrix.bytes.len };
    const p: MatvecRowsParams = .{ .columns = @intCast(matrix.columns), .encoding = matrix.encoding, .stride = @intCast(stride), .rows = @intCast(matrix.rows), .tokens = @intCast(tokens), .in_stride = @intCast(in_stride), .out_stride = @intCast(out_stride) };
    if (!self.generic_only) if (specializedMatvecRows(matrix.encoding, tokens, weights.offset, stride, input.offset)) |kernel| {
        const rows_per_group = matvec_rows_per_simdgroup * simdgroups_per_matvec_group;
        try self.dispatch(kernel, &.{ weights, input, output }, p, @intCast(@divCeil(matrix.rows, rows_per_group)), 32 * simdgroups_per_matvec_group, shape);
        return;
    };
    try self.dispatch(.matvec_rows, &.{ weights, input, output }, p, @intCast(matrix.rows), 32, shape);
}
/// The multi-row bodies that exist (Q4_K, Q5_K, Q6_K, IQ4_XS), one per token
/// count 2..8; every other encoding takes the generic `nu_matvec_rows` above.
/// The token count is a template parameter so the body's accumulator loops
/// are compile-time bound (see kernels/).
pub fn specializedMatvecRows(encoding: u32, tokens: usize, weight_offset: usize, stride: usize, input_offset: usize) ?Kernel {
    if (input_offset % 16 != 0 or tokens < 2 or tokens > matvec_rows_max or !blockAligned(encoding, weight_offset, stride)) return null;
    const bodies = switch (encoding) {
        12 => comptime matvecRowsBodies("q4_k"),
        13 => comptime matvecRowsBodies("q5_k"),
        14 => comptime matvecRowsBodies("q6_k"),
        23 => comptime matvecRowsBodies("iq4_xs"),
        16 => comptime matvecRowsBodies("iq2_xxs"),
        else => return null,
    };
    return bodies[tokens - 2];
}
/// `matvec_rows_<name>_t2` .. `_t<matvec_rows_max>`, looked up by name.
fn matvecRowsBodies(comptime name: []const u8) [matvec_rows_max - 1]Kernel {
    var bodies: [matvec_rows_max - 1]Kernel = undefined;
    for (&bodies, 2..) |*body, tokens| body.* = @field(Kernel, std.fmt.comptimePrint("matvec_rows_{s}_t{d}", .{ name, tokens }));
    return bodies;
}
/// Picks a specialized matvec when its vector loads are aligned
/// (`blockAligned`) and the input vector is float4-aligned. Otherwise
/// `null`: generic path. The K-quant kernels walk 256-value strides;
/// the Q4_0, Q5_0, and Q5_1 kernels walk 32-value blocks and the ternary
/// kernels 128-value blocks (lanes past the last block idle), so any
/// whole-block row serves those five.
pub fn specializedMatvec(encoding: u32, weight_offset: usize, stride: usize, input_offset: usize) ?Kernel {
    if (input_offset % 16 != 0 or !blockAligned(encoding, weight_offset, stride)) return null;
    return switch (encoding) {
        2 => .matvec_q4_0,
        6 => .matvec_q5_0,
        7 => .matvec_q5_1,
        142 => .matvec_pq2_0,
        143 => .matvec_ptq1_0,
        11 => .matvec_q3_k,
        21 => .matvec_iq3_s,
        12 => .matvec_q4_k,
        13 => .matvec_q5_k,
        14 => .matvec_q6_k,
        23 => .matvec_iq4_xs,
        16 => .matvec_iq2_xxs,
        17 => .matvec_iq2_xs,
        22 => .matvec_iq2_s,
        18 => .matvec_iq3_xxs,
        10 => .matvec_q2_k,
        19 => .matvec_iq1_s,
        29 => .matvec_iq1_m,
        else => unreachable,
    };
}
/// Borrowed projection descriptors. All projections read the same input.
/// Plain outputs must be disjoint; a gate pair (SiLU or tanh GELU) names
/// the same output in both entries and writes only act(segment[0]) * segment[1].
pub const Segment = struct { weights: Buffer, matrix: cpu.Matrix, output: Buffer };
pub const SegmentMode = enum(u32) { plain, silu_mul_pair, gelu_mul_pair };
const SegmentParams = extern struct { rows: u32, weight_slot: u32, weight_offset: u32, encoding: u32, stride: u32, output_slot: u32, output_offset: u32 };
const MatvecSegments = extern struct { columns: u32, blocks: u32, count: u32, mode: u32, splits: u32, split_blocks: u32, total_rows: u32, partials_slot: u32, segments: [4]SegmentParams };

/// Bind each underlying buffer once at offset zero. Segment byte offsets
/// retain its original slice. This fits both four weights + packed DeltaNet
/// outputs and three weights + attention outputs in six slots, plus input.
/// The rebased descriptors are used only as Metal bindings, never as host views.
fn segmentSlot(bindings: *[7]Buffer, used: *usize, buffer: Buffer) !u32 {
    for (1..used.*) |i| if (bindings[i].id == buffer.id) return @intCast(i);
    if (used.* == bindings.len) return error.InvalidShape;
    const index = used.*;
    bindings[index] = buffer;
    bindings[index].offset = 0;
    used.* += 1;
    return @intCast(index);
}
/// Records the merged matvec of up to four projections in one dispatch.
/// `error.MergeDeclined` (nothing recorded) when a segment's specialized
/// kernel runs faster alone (`mergeDeclined`); the caller then runs the
/// segments as standalone matvecs and a pair's activation.
pub fn matvecSegments(self: *Backend, segments: []const Segment, input: Buffer, mode: SegmentMode) !void {
    return matvecSegmentsImpl(self, segments, input, mode, null);
}
/// The seven 2-bit encodings' standalone kernels beat their bodies inside
/// the merged kernel by 11–23 % on the gate+up pair, even with the two
/// extra dispatches and `siluMul` (docs/engine/apple-gpu.md § Saluki's
/// 2-bit matvecs on the `ffn_down` shape), so a merge holding one is declined.
pub fn mergeDeclined(kernel: Kernel) bool {
    return switch (kernel) {
        .matvec_iq2_xxs, .matvec_iq2_xs, .matvec_iq2_s, .matvec_iq3_xxs, .matvec_q2_k, .matvec_iq1_s, .matvec_iq1_m => true,
        else => false,
    };
}
/// `matvecSegments` forcing the split count (the split sweep's knob; 1 is
/// the single-pass kernel). Only plain-mode merges whose every segment has
/// a specialized split body (Q4_K/Q5_K) are served. As with
/// `matvecSplits`, no production merge routes here (the split measured
/// behind the single pass); the segment split kernels stay as the measured fixture.
pub fn matvecSegmentsSplits(self: *Backend, segments: []const Segment, input: Buffer, mode: SegmentMode, splits: usize) !void {
    if (splits == 0 or splits > split_k_max) return error.InvalidShape;
    return matvecSegmentsImpl(self, segments, input, mode, splits);
}
fn matvecSegmentsImpl(self: *Backend, segments: []const Segment, input: Buffer, mode: SegmentMode, splits_override: ?usize) !void {
    // Shape and the decline first, then availability: with Metal compiled
    // out the early `MetalNotEnabled` return would otherwise make
    // `InvalidShape` and `MergeDeclined` unreachable, and the adapters'
    // switch on them would not compile in a CPU-only build. The decline
    // needs no validated shape: a declined merge's segments are validated
    // by the standalone matvecs that run them.
    if (segments.len == 0 or segments.len > 4) return error.InvalidShape;
    if (splits_override == null and !self.generic_only) for (segments) |s| {
        if (s.matrix.rows == 0) continue;
        const kernel = specializedMatvec(s.matrix.encoding, s.weights.offset, s.matrix.bytes.len / s.matrix.rows, input.offset) orelse continue;
        if (mergeDeclined(kernel)) return error.MergeDeclined;
    };
    if (!enabled) return error.MetalNotEnabled;
    const columns = segments[0].matrix.columns;
    if (columns == 0 or columns > std.math.maxInt(u32) / 4 or columns % 16 != 0 or input.len < columns * 4 or input.offset % 4 != 0) return error.InvalidShape;
    if (mode != .plain and (segments.len != 2 or segments[0].matrix.rows != segments[1].matrix.rows or segments[0].output.id != segments[1].output.id or segments[0].output.offset != segments[1].output.offset)) return error.InvalidShape;
    var bindings: [7]Buffer = @splat(input);
    var used: usize = 1;
    var p: MatvecSegments = std.mem.zeroes(MatvecSegments);
    p.columns = @intCast(columns);
    p.blocks = @intCast(columns / 256);
    p.count = @intCast(segments.len);
    p.mode = @backingInt(mode);
    var total_rows: u32 = 0;
    var bytes: u64 = 0;
    // The forced split path needs every segment specialized *with a split
    // body* (neither the generic body nor the non-split encodings honour a
    // K range) and a free binding slot for the row partials.
    var all_split_capable = mode == .plain;
    for (segments, 0..) |s, i| {
        const m = s.matrix;
        if (m.rows == 0 or m.rows > std.math.maxInt(u32) / 4 or m.rows % 16 != 0 or m.columns != columns or m.bytes.len % m.rows != 0) return error.InvalidShape;
        const stride = m.bytes.len / m.rows;
        try quant.validateRow(m.encoding, stride, columns);
        if (s.weights.len < m.bytes.len or s.output.len < m.rows * 4 or s.output.offset % 4 != 0) return error.InvalidShape;
        if (Buffer.overlaps(s.output, m.rows * 4, input, columns * 4)) return error.InvalidShape;
        for (segments) |other| if (Buffer.overlaps(s.output, m.rows * 4, other.weights, other.matrix.bytes.len)) return error.InvalidShape;
        if (mode == .plain) for (segments[0..i]) |other| {
            if (Buffer.overlaps(s.output, m.rows * 4, other.output, other.matrix.rows * 4)) return error.InvalidShape;
        };
        total_rows = std.math.add(u32, total_rows, @intCast(m.rows)) catch return error.InvalidShape;
        bytes = std.math.add(u64, bytes, m.bytes.len) catch return error.InvalidShape;
        const kernel = if (self.generic_only) null else specializedMatvec(m.encoding, s.weights.offset, stride, input.offset);
        const specialized = kernel != null;
        all_split_capable = all_split_capable and (if (kernel) |k| matvecSplitKernel(k) != null else false);
        p.segments[i] = .{
            .rows = @intCast(m.rows),
            .weight_slot = try segmentSlot(&bindings, &used, s.weights),
            .weight_offset = std.math.cast(u32, s.weights.offset) orelse return error.InvalidShape,
            .encoding = m.encoding | (if (specialized) @as(u32, 0) else 0x80000000),
            .stride = std.math.cast(u32, stride) orelse return error.InvalidShape,
            .output_slot = try segmentSlot(&bindings, &used, s.output),
            .output_offset = std.math.cast(u32, s.output.offset) orelse return error.InvalidShape,
        };
    }
    // Unused slots bind the valid input buffer, because the kernel signature
    // declares all seven. No dispatch is recorded before all validation passes.
    const splits = splits_override orelse 1;
    if (splits > 1) {
        if (!all_split_capable or used >= bindings.len) return error.InvalidShape;
        const scratch = try splitScratch(self);
        if (total_rows * splits * 4 > scratch.len) return error.InvalidShape;
        bindings[used] = scratch;
        p.splits = @intCast(splits);
        p.split_blocks = @intCast(@divCeil(p.blocks, splits));
        p.total_rows = total_rows;
        p.partials_slot = @intCast(used);
        const groups: u32 = @intCast((total_rows / 16) * splits);
        try self.dispatch(.matvec_segments_split, &bindings, p, groups, 128, .{ .rows = total_rows, .columns = p.columns, .bytes = bytes * splits });
        try self.dispatch(.segment_reduce_splits, &bindings, p, perElement(total_rows), 256, .{});
        return;
    }
    const groups = if (mode == .plain) total_rows / 16 else p.segments[0].rows / 8;
    try self.dispatch(.matvec_segments, &bindings, p, groups, 128, .{ .rows = total_rows, .columns = p.columns, .bytes = bytes });
}
