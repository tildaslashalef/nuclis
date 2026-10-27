//! Causal attention: single position, flash-decoding, verify, prefill chunks, and the KV cache's packing: the host side of `kernels/attention.metal`. Methods of `Backend`,
//! which `root.zig` aliases so callers write `backend.attention(…)`.
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

pub const AttentionParams = extern struct { query_heads: u32, kv_heads: u32, key_width: u32, value_width: u32, visible: u32, scale: f32 };
/// `precision` is how `keys`/`values` are stored; queries, scores,
/// and the output are always F32. This three-pass path reads F32 or
/// F16; `attentionDecode` also reads Q8_0.
pub const AttentionShape = struct { query_heads: usize, kv_heads: usize, key_width: usize, value_width: usize, visible: usize, scale: f32, precision: Precision = .f32 };
/// Three dispatches; `scores` needs query_heads * visible floats.
pub fn attention(self: *Backend, keys: Buffer, values: Buffer, queries: Buffer, scores: Buffer, output: Buffer, s: AttentionShape) !void {
    if (s.query_heads == 0 or s.kv_heads == 0 or s.query_heads % s.kv_heads != 0 or s.key_width == 0 or s.value_width == 0 or s.visible == 0 or !std.math.isFinite(s.scale) or s.scale <= 0) return error.InvalidShape;
    if (s.precision == .q8_0) return error.InvalidShape;
    try checkCache(s.precision, keys, values, s.visible * s.kv_heads, s.key_width, s.value_width);
    if (queries.len < s.query_heads * s.key_width * 4 or scores.len < s.query_heads * s.visible * 4 or output.len < s.query_heads * s.value_width * 4) return error.InvalidShape;
    const p: AttentionParams = .{ .query_heads = @intCast(s.query_heads), .kv_heads = @intCast(s.kv_heads), .key_width = @intCast(s.key_width), .value_width = @intCast(s.value_width), .visible = @intCast(s.visible), .scale = s.scale };
    const half = s.precision == .f16;
    try self.dispatch(if (half) .attention_scores_h else .attention_scores, &.{ keys, queries, scores }, p, @intCast(s.query_heads * s.visible), 32, .{});
    try self.dispatch(.attention_softmax, &.{scores}, p, @intCast(s.query_heads), 32, .{});
    try self.dispatch(if (half) .attention_values_h else .attention_values, &.{ values, scores, output }, p, @intCast(s.query_heads * s.value_width), 32, .{});
}
pub const AttentionDecodeParams = extern struct { query_heads: u32, kv_heads: u32, key_width: u32, value_width: u32, visible: u32, splits: u32, scale: f32, rows: u32, q_stride: u32, out_stride: u32, window: u32 };
/// Most splits of the visible range one `attentionDecode` uses, and the
/// rows each split holds before another split is added; must match the
/// partial layout the kernels write.
pub const decode_splits_max = 64;
pub const decode_split_rows = 256;
pub fn attentionDecodeSplits(visible: usize) usize {
    return @min(decode_splits_max, @divCeil(visible, decode_split_rows));
}
/// Floats the partial buffer must hold: `[query_heads][splits][2 + value_width]`.
pub fn attentionDecodePartials(query_heads: usize, value_width: usize) usize {
    return query_heads * decode_splits_max * (2 + value_width);
}
/// Query heads one threadgroup of the split pass covers: 8 in the
/// instantiation for widths up to 256, 4 in the wide one (widths up to
/// 512); a KV head with more query heads takes several threadgroups
/// per split, each reading the slice once.
pub fn attentionDecodeHeadGroups(query_heads: usize, kv_heads: usize, key_width: usize, value_width: usize) usize {
    const per_group: usize = if (key_width > 256 or value_width > 256) 4 else 8;
    const g = query_heads / kv_heads;
    return @divCeil(g, per_group);
}
/// How the split pass runs a shape: the kernel, its threadgroups
/// per (KV head, split, row), and its threads per threadgroup.
const DecodePlan = struct { kernel: Kernel, groups_per_split: usize, threads: u32 };
/// The head-split instantiation (`nu_attention_heads_t`) when keys and
/// values share a width of 128, 256, or 512 and the KV head's query
/// heads fit one threadgroup; the general kernel otherwise. Query heads
/// per SIMD group and preferred row lanes per (precision, width) are
/// the measured optimum (docs/engine/metal-backend.md § The split pass
/// at the memory roof); row lanes halve until the threadgroup holds
/// at most 1,024 threads and their merge fits the kernel's 4,096-float
/// stage. The group counts must match the kernel's instantiations.
fn decodePlan(precision: Precision, query_heads: usize, kv_heads: usize, key_width: usize, value_width: usize) DecodePlan {
    const general: DecodePlan = .{ .kernel = decodeKernel(precision, key_width, value_width), .groups_per_split = attentionDecodeHeadGroups(query_heads, kv_heads, key_width, value_width), .threads = 128 };
    if (key_width != value_width) return general;
    const Choice = struct { kernel: Kernel, group: usize, lanes: usize };
    const choice: Choice = switch (key_width) {
        128 => switch (precision) {
            .f32 => .{ .kernel = .attention_heads_f32_128, .group = 4, .lanes = 2 },
            .f16 => .{ .kernel = .attention_heads_f16_128, .group = 4, .lanes = 2 },
            .q8_0 => .{ .kernel = .attention_heads_q8_128, .group = 4, .lanes = 2 },
        },
        256 => switch (precision) {
            .f32 => .{ .kernel = .attention_heads_f32_256, .group = 2, .lanes = 2 },
            .f16 => .{ .kernel = .attention_heads_f16_256, .group = 2, .lanes = 2 },
            .q8_0 => .{ .kernel = .attention_heads_q8_256, .group = 3, .lanes = 4 },
        },
        512 => switch (precision) {
            .f32 => .{ .kernel = .attention_heads_f32_512, .group = 1, .lanes = 2 },
            .f16 => .{ .kernel = .attention_heads_f16_512, .group = 1, .lanes = 2 },
            .q8_0 => .{ .kernel = .attention_heads_q8_512, .group = 1, .lanes = 2 },
        },
        else => return general,
    };
    const head_groups = @divCeil(query_heads / kv_heads, choice.group);
    if (head_groups * 32 > 1024) return general;
    var lanes = choice.lanes;
    while (lanes > 1 and (head_groups * lanes * 32 > 1024 or head_groups * choice.group * key_width > 4096)) lanes /= 2;
    return .{ .kernel = choice.kernel, .groups_per_split = 1, .threads = @intCast(head_groups * lanes * 32) };
}
/// Flash-decoding attention: two dispatches (split pass, merge)
/// and no score buffer; `partials` needs `attentionDecodePartials`
/// floats. The shape contract of `attention`, plus widths of at most
/// 512 (above 256 through the wide instantiation); the cache may also
/// be `q8_0`, whose widths are multiples of 32 (a block per head row).
pub fn attentionDecode(self: *Backend, keys: Buffer, values: Buffer, queries: Buffer, partials: Buffer, output: Buffer, s: AttentionShape) !void {
    if (s.query_heads == 0 or s.kv_heads == 0 or s.query_heads % s.kv_heads != 0 or s.key_width == 0 or s.key_width > 512 or s.value_width == 0 or s.value_width > 512 or s.visible == 0 or !std.math.isFinite(s.scale) or s.scale <= 0) return error.InvalidShape;
    const splits = attentionDecodeSplits(s.visible);
    const plan = decodePlan(s.precision, s.query_heads, s.kv_heads, s.key_width, s.value_width);
    try checkCache(s.precision, keys, values, s.visible * s.kv_heads, s.key_width, s.value_width);
    if (queries.len < s.query_heads * s.key_width * 4 or partials.len < s.query_heads * splits * (2 + s.value_width) * 4 or output.len < s.query_heads * s.value_width * 4) return error.InvalidShape;
    const p: AttentionDecodeParams = .{ .query_heads = @intCast(s.query_heads), .kv_heads = @intCast(s.kv_heads), .key_width = @intCast(s.key_width), .value_width = @intCast(s.value_width), .visible = @intCast(s.visible), .splits = @intCast(splits), .scale = s.scale, .rows = 1, .q_stride = @intCast(s.query_heads * s.key_width), .out_stride = @intCast(s.query_heads * s.value_width), .window = 0 };
    // Cache traffic per dispatch, for the profile: every row once.
    const bytes: u64 = @intCast(s.visible * s.precision.bytes(s.kv_heads * (s.key_width + s.value_width)));
    try self.dispatch(plan.kernel, &.{ keys, values, queries, partials }, p, @intCast(s.kv_heads * plan.groups_per_split * splits), plan.threads, .{ .bytes = bytes });
    try self.dispatch(.attention_merge, &.{ partials, output }, p, @intCast(s.query_heads), 256, .{});
}
/// The most rows `attentionVerify` takes: a verify batch or a drafter's
/// commit, never a prefill chunk.
pub const verify_max_rows = 16;
/// Floats `attentionVerify`'s partial buffer must hold for `rows` rows.
pub fn attentionVerifyPartials(rows: usize, query_heads: usize, value_width: usize) usize {
    return rows * attentionDecodePartials(query_heads, value_width);
}
/// Few-query causal attention: the chunk contract of `attentionChunk`
/// (row t sees the cache rows `[0, position + t]`) for at most
/// `verify_max_rows` rows, through the flash-decoding split pass, so the
/// grid is (KV head, split, row) instead of one threadgroup per query
/// head. Queries are F32 whatever the cache's `precision`, which may be
/// `q8_0` (as in `attentionDecode`); `partials` needs
/// `attentionVerifyPartials` floats.
/// A window as in `attentionChunk`; no span.
pub fn attentionVerify(self: *Backend, keys: Buffer, values: Buffer, queries: Buffer, partials: Buffer, output: Buffer, s: AttentionChunkShape) !void {
    if (s.count > verify_max_rows or s.span.end != 0 or s.key_width > 512) return error.InvalidShape;
    try checkAttentionChunk(s, keys, values, queries, output, 4);
    const visible = s.position + s.count;
    const splits = attentionDecodeSplits(visible);
    const plan = decodePlan(s.precision, s.query_heads, s.kv_heads, s.key_width, s.value_width);
    if (partials.len < s.count * s.query_heads * splits * (2 + s.value_width) * 4) return error.InvalidShape;
    const p: AttentionDecodeParams = .{ .query_heads = @intCast(s.query_heads), .kv_heads = @intCast(s.kv_heads), .key_width = @intCast(s.key_width), .value_width = @intCast(s.value_width), .visible = @intCast(visible), .splits = @intCast(splits), .scale = s.scale, .rows = @intCast(s.count), .q_stride = @intCast(s.q_stride), .out_stride = @intCast(s.out_stride), .window = @intCast(s.window) };
    const bytes: u64 = @intCast(s.count * visible * s.precision.bytes(s.kv_heads * (s.key_width + s.value_width)));
    try self.dispatch(plan.kernel, &.{ keys, values, queries, partials }, p, @intCast(s.kv_heads * plan.groups_per_split * splits * s.count), plan.threads, .{ .bytes = bytes });
    try self.dispatch(.attention_merge, &.{ partials, output }, p, @intCast(s.query_heads * s.count), 256, .{});
}
pub const PackParams = extern struct { count0: u32, count1: u32 };
/// One F32 → F16 conversion: `count` floats of `src` into `dst`.
pub const Pack = struct { dst: Buffer, src: Buffer, count: usize };
/// Rounds one or two F32 vectors to F16 in one dispatch (round to
/// nearest even, as `@floatCast`): the attention projections stay F32
/// and this writes the F16 cache slot. Destinations must not overlap
/// their sources.
pub fn packHalf(self: *Backend, pairs: []const Pack) !void {
    if (pairs.len == 0 or pairs.len > 2) return error.InvalidShape;
    var total: usize = 0;
    for (pairs) |pair| {
        if (pair.count == 0 or pair.dst.len < pair.count * 2 or pair.src.len < pair.count * 4 or pair.dst.offset % 2 != 0 or pair.src.offset % 4 != 0) return error.InvalidShape;
        if (Buffer.overlaps(pair.dst, pair.count * 2, pair.src, pair.count * 4)) return error.InvalidShape;
        total += pair.count;
    }
    const second = if (pairs.len == 2) pairs[1] else pairs[0];
    const p: PackParams = .{ .count0 = @intCast(pairs[0].count), .count1 = if (pairs.len == 2) @intCast(pairs[1].count) else 0 };
    try self.dispatch(.pack_half, &.{ pairs[0].dst, pairs[0].src, second.dst, second.src }, p, perElement(total), 256, .{});
}
/// Quantizes one or two F32 vectors to Q8_0 blocks in one dispatch, the
/// arithmetic of `quant_encode.q8_0Row` (bit-identical): the `q8_0`
/// cache slot's write. Counts are multiples of 32; destinations must
/// not overlap their sources.
pub fn packQ8(self: *Backend, pairs: []const Pack) !void {
    try packPairs(self, .pack_q8_0, pairs, .q8_0, .f32);
}
/// Expands one or two Q8_0 vectors (`count` values each) to F16: the
/// prefill chunk's half view of a `q8_0` cache.
pub fn unpackQ8(self: *Backend, pairs: []const Pack) !void {
    try packPairs(self, .unpack_q8_0, pairs, .f16, .q8_0);
}
/// Writes F32 rows into a cache region at `precision`: a copy, an F16
/// pack, or a Q8_0 pack, so a plan's cache write is one call.
pub fn storeRows(self: *Backend, pairs: []const Pack, precision: Precision) !void {
    switch (precision) {
        .f32 => for (pairs) |pair| try self.copy(pair.dst, pair.src, pair.count),
        .f16 => try self.packHalf(pairs),
        .q8_0 => try self.packQ8(pairs),
    }
}
fn packPairs(self: *Backend, kernel: Kernel, pairs: []const Pack, dst: Precision, src: Precision) !void {
    if (pairs.len == 0 or pairs.len > 2) return error.InvalidShape;
    var total: usize = 0;
    for (pairs) |pair| {
        if (pair.count == 0 or pair.count % Precision.q8_block != 0) return error.InvalidShape;
        const dst_bytes = dst.bytes(pair.count);
        const src_bytes = src.bytes(pair.count);
        if (pair.dst.len < dst_bytes or pair.src.len < src_bytes or pair.dst.offset % cacheAlignment(dst) != 0 or pair.src.offset % cacheAlignment(src) != 0) return error.InvalidShape;
        if (Buffer.overlaps(pair.dst, dst_bytes, pair.src, src_bytes)) return error.InvalidShape;
        total += pair.count;
    }
    const second = if (pairs.len == 2) pairs[1] else pairs[0];
    const p: PackParams = .{ .count0 = @intCast(pairs[0].count), .count1 = if (pairs.len == 2) @intCast(pairs[1].count) else 0 };
    try self.dispatch(kernel, &.{ pairs[0].dst, pairs[0].src, second.dst, second.src }, p, perElement(total), 256, .{});
}
/// Byte alignment a binding of `precision` values needs (Q8_0 blocks
/// are 2-byte aligned, by their F16 scale).
fn cacheAlignment(precision: Precision) usize {
    return switch (precision) {
        .f32 => 4,
        .f16, .q8_0 => 2,
    };
}
/// The split-pass instantiation for a cache precision and widths.
fn decodeKernel(precision: Precision, key_width: usize, value_width: usize) Kernel {
    const wide = key_width > 256 or value_width > 256;
    return switch (precision) {
        .f32 => if (wide) .attention_decode_w else .attention_decode,
        .f16 => if (wide) .attention_decode_wh else .attention_decode_h,
        .q8_0 => if (wide) .attention_decode_wq8 else .attention_decode_q8,
    };
}
/// `rows` rows of keys and values at `precision`: lengths, alignment,
/// and, for `q8_0`, widths of whole blocks.
fn checkCache(precision: Precision, keys: Buffer, values: Buffer, rows: usize, key_width: usize, value_width: usize) !void {
    if (!precision.fits(key_width) or !precision.fits(value_width)) return error.InvalidShape;
    if (keys.len < rows * precision.bytes(key_width) or values.len < rows * precision.bytes(value_width)) return error.InvalidShape;
    if (keys.offset % cacheAlignment(precision) != 0 or values.offset % cacheAlignment(precision) != 0) return error.InvalidShape;
}
pub const AttentionChunkParams = extern struct { query_heads: u32, kv_heads: u32, key_width: u32, value_width: u32, position: u32, count: u32, q_stride: u32, out_stride: u32, scale: f32, window: u32, span_begin: u32, span_end: u32 };
/// Chunk rows `[begin, end)` that attend bidirectionally among
/// themselves: each sees keys up to `position + end` (exclusive) instead
/// of its own row, the window still bounding it from below. Gemma 4's
/// image span on sliding layers; an empty span is plain causal.
pub const Span = struct { begin: usize = 0, end: usize = 0 };
/// `count` query rows at cache positions `position..position + count`,
/// each attending causally over the cache rows `[0, position + t]`, or
/// `[position + t + 1 - window, position + t]` when `window` is nonzero
/// (a sliding window; the caller may slice the cache so that row 0
/// is the earliest key any row of the chunk can see).
/// `queries` holds `[row][q_stride]` with the row's heads at
/// `head * key_width`; `output` is written `[row][out_stride]` likewise.
/// `precision` is how the cache **and the queries** are stored:
/// the `f16` instantiation multiplies half operands into F32
/// accumulators, so the caller packs its queries to half first
/// (`packHalf`); the output is F32 either way. The chunk bodies read
/// no `q8_0` cache: the caller expands it to F16 first (`unpackQ8`).
pub const AttentionChunkShape = struct { query_heads: usize, kv_heads: usize, key_width: usize, value_width: usize, position: usize, count: usize, q_stride: usize, out_stride: usize, scale: f32, precision: Precision = .f32, window: usize = 0, span: Span = .{} };
/// The most rows one chunk dispatch covers: a projector's whole patch
/// set (10,080 for Gemma 4's largest image) runs as one span.
pub const max_chunk_rows = 16384;
/// Rows the query and output buffers must hold for `count` rows: the
/// last SIMD-group tile computes and stores on the caller's padding.
pub fn attentionChunkRows(count: usize) usize {
    return (count + 7) / 8 * 8;
}
/// One dispatch per layer for a prefill chunk: threadgroup per
/// (query head, 32-query tile, 256 value columns), no score buffer.
/// Reads at most `position + count` cache rows. Value widths above 256
/// (up to 512) take one threadgroup per 256 columns. Counts at or below
/// `attention_reuse_max_rows` take the register-reuse body instead.
pub fn attentionChunk(self: *Backend, keys: Buffer, values: Buffer, queries: Buffer, output: Buffer, s: AttentionChunkShape) !void {
    if (s.precision == .q8_0) return error.InvalidShape;
    try checkAttentionChunk(s, keys, values, queries, output, s.precision.bytes(1));
    const p = attentionChunkParams(s);
    const tiles = @divCeil(s.count, 32);
    const value_splits = (s.value_width + 255) / 256;
    // The register-reuse body has no span; a span takes the row-split body.
    const reuse = s.value_width == 256 and s.count <= self.attention_reuse_max_rows and s.span.end == 0;
    const half = s.precision == .f16;
    const kernel: Kernel = if (reuse) (if (half) .attention_chunk_reuse_h else .attention_chunk_reuse) else if (half) .attention_chunk_h else .attention_chunk;
    try self.dispatch(kernel, &.{ keys, values, queries, output }, p, @intCast(s.query_heads * tiles * value_splits), 128, .{});
}
/// The register-reuse body explicitly: the same contract, but each
/// threadgroup's four SIMD groups split the value columns instead of the
/// query rows, so the published P tile makes a V block serve four
/// multiplies. Measured against `attentionChunk` by `--attention-bench`.
pub fn attentionChunkReuse(self: *Backend, keys: Buffer, values: Buffer, queries: Buffer, output: Buffer, s: AttentionChunkShape) !void {
    if (s.precision == .q8_0) return error.InvalidShape;
    try checkAttentionChunk(s, keys, values, queries, output, s.precision.bytes(1));
    if (s.span.end != 0) return error.InvalidShape;
    const p = attentionChunkParams(s);
    const tiles = @divCeil(s.count, 32);
    const value_splits = (s.value_width + 255) / 256;
    try self.dispatch(if (s.precision == .f16) .attention_chunk_reuse_h else .attention_chunk_reuse, &.{ keys, values, queries, output }, p, @intCast(s.query_heads * tiles * value_splits), 128, .{});
}
fn attentionChunkParams(s: AttentionChunkShape) AttentionChunkParams {
    return .{ .query_heads = @intCast(s.query_heads), .kv_heads = @intCast(s.kv_heads), .key_width = @intCast(s.key_width), .value_width = @intCast(s.value_width), .position = @intCast(s.position), .count = @intCast(s.count), .q_stride = @intCast(s.q_stride), .out_stride = @intCast(s.out_stride), .scale = s.scale, .window = @intCast(s.window), .span_begin = @intCast(s.span.begin), .span_end = @intCast(s.span.end) };
}
/// `query_elem` is the queries' bytes per value: the cache's for the
/// chunk bodies, 4 for `attentionVerify`.
fn checkAttentionChunk(s: AttentionChunkShape, keys: Buffer, values: Buffer, queries: Buffer, output: Buffer, query_elem: usize) !void {
    if (s.query_heads == 0 or s.kv_heads == 0 or s.query_heads % s.kv_heads != 0 or !std.math.isFinite(s.scale) or s.scale <= 0) return error.InvalidShape;
    if (s.key_width == 0 or s.key_width % 8 != 0 or s.value_width == 0 or s.value_width % 8 != 0 or s.value_width > 512) return error.InvalidShape;
    if (s.count == 0 or s.count > max_chunk_rows or s.position > 32768 - s.count) return error.InvalidShape;
    if (s.span.begin > s.span.end or s.span.end > s.count) return error.InvalidShape;
    if (s.q_stride < s.query_heads * s.key_width or s.out_stride < s.query_heads * s.value_width) return error.InvalidShape;
    const total = s.position + s.count;
    const rows = attentionChunkRows(s.count);
    try checkCache(s.precision, keys, values, total * s.kv_heads, s.key_width, s.value_width);
    if (queries.len < rows * s.q_stride * query_elem or output.len < rows * s.out_stride * 4) return error.InvalidShape;
    if (queries.offset % query_elem != 0) return error.InvalidShape;
}
