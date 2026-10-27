//! The encoders' kernels: LayerNorm, bias rows, full and segment attention: the host side of `kernels/encoders.metal`. Methods of `Backend`,
//! which `root.zig` aliases so callers write `backend.layerNorm(…)`.
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
const Norm = Backend.Norm;
const CountParams = Backend.CountParams;
const attentionChunkRows = Backend.attentionChunkRows;

pub const LayerNormParams = extern struct { width: u32, in_stride: u32, out_stride: u32, eps: f32 };
/// LayerNorm with weight and bias over `rows` rows of `width`
/// (`spec.silu_multiplier` unused); input and output may alias exactly.
pub fn layerNorm(self: *Backend, input: Buffer, weight: Buffer, bias: Buffer, output: Buffer, spec: Norm) !void {
    if (spec.rows == 0 or spec.width == 0 or spec.in_stride < spec.width or spec.out_stride < spec.width or !(spec.eps > 0)) return error.InvalidShape;
    if (input.len < ((spec.rows - 1) * spec.in_stride + spec.width) * 4 or output.len < ((spec.rows - 1) * spec.out_stride + spec.width) * 4 or weight.len < spec.width * 4 or bias.len < spec.width * 4) return error.InvalidShape;
    const p: LayerNormParams = .{ .width = @intCast(spec.width), .in_stride = @intCast(spec.in_stride), .out_stride = @intCast(spec.out_stride), .eps = spec.eps };
    try self.dispatch(.layernorm, &.{ input, weight, bias, output }, p, @intCast(spec.rows), 256, .{});
}
pub const BiasRowsParams = extern struct { width: u32, rows: u32, stride: u32 };
/// `x[row · stride + i] += bias[i]` over `rows` rows of `width`.
pub fn addBiasRows(self: *Backend, x: Buffer, bias: Buffer, width: usize, rows: usize, stride: usize) !void {
    if (width == 0 or rows == 0 or stride < width or x.len < ((rows - 1) * stride + width) * 4 or bias.len < width * 4) return error.InvalidShape;
    const p: BiasRowsParams = .{ .width = @intCast(width), .rows = @intCast(rows), .stride = @intCast(stride) };
    try self.dispatch(.add_bias_rows, &.{ x, bias }, p, perElement(width * rows), 256, .{});
}
/// x = gelu(x) (the tanh form of `cpu.gelu`) over `count` values.
pub fn gelu(self: *Backend, x: Buffer, count: usize) !void {
    if (count == 0 or x.len < count * 4) return error.InvalidShape;
    try self.dispatch(.gelu_inplace, &.{x}, CountParams{ .count = @intCast(count) }, perElement(count), 256, .{});
}
pub const AttentionFullParams = extern struct { heads: u32, width: u32, rows: u32, q_stride: u32, kv_stride: u32, out_stride: u32, scale: f32 };
pub const AttentionFullShape = struct { heads: usize, width: usize, rows: usize, q_stride: usize, kv_stride: usize, out_stride: usize, scale: f32 };
/// Maximum rows one bidirectional attention can cover (threadgroup memory).
pub const attention_full_max_rows = 4096;
/// Bidirectional attention over `rows` rows: every row attends to every
/// row, `heads` heads of `width` (at most 256) per row. Queries, keys, and
/// values are separate buffers (or slices of one) with their own row
/// strides; the output is `heads · width` wide at `out_stride`.
pub fn attentionFull(self: *Backend, queries: Buffer, keys: Buffer, values: Buffer, output: Buffer, s: AttentionFullShape) !void {
    if (s.heads == 0 or s.width == 0 or s.width > 256 or s.rows == 0 or s.rows > attention_full_max_rows) return error.InvalidShape;
    const row_floats = s.heads * s.width;
    if (s.q_stride < row_floats or s.kv_stride < row_floats or s.out_stride < row_floats) return error.InvalidShape;
    if (queries.len < ((s.rows - 1) * s.q_stride + row_floats) * 4 or keys.len < ((s.rows - 1) * s.kv_stride + row_floats) * 4 or values.len < ((s.rows - 1) * s.kv_stride + row_floats) * 4 or output.len < ((s.rows - 1) * s.out_stride + row_floats) * 4) return error.InvalidShape;
    if (Buffer.overlaps(output, ((s.rows - 1) * s.out_stride + row_floats) * 4, queries, ((s.rows - 1) * s.q_stride + row_floats) * 4) or Buffer.overlaps(output, ((s.rows - 1) * s.out_stride + row_floats) * 4, keys, ((s.rows - 1) * s.kv_stride + row_floats) * 4) or Buffer.overlaps(output, ((s.rows - 1) * s.out_stride + row_floats) * 4, values, ((s.rows - 1) * s.kv_stride + row_floats) * 4)) return error.InvalidShape;
    const p: AttentionFullParams = .{ .heads = @intCast(s.heads), .width = @intCast(s.width), .rows = @intCast(s.rows), .q_stride = @intCast(s.q_stride), .kv_stride = @intCast(s.kv_stride), .out_stride = @intCast(s.out_stride), .scale = s.scale };
    try self.dispatch(.attention_full, &.{ queries, keys, values, output }, p, @intCast(s.rows * s.heads), 256, .{});
}
pub const AttentionSegmentsParams = extern struct { heads: u32, width: u32, rows: u32, q_stride: u32, kv_stride: u32, out_stride: u32, scale: f32, window: u32 };
pub const AttentionSegmentsShape = struct {
    heads: usize,
    width: usize,
    rows: usize,
    q_stride: usize,
    kv_stride: usize,
    out_stride: usize,
    scale: f32,
    /// Row i sees key j when |i − j| ≤ window; null sees its whole sequence.
    window: ?usize = null,
};
/// `attentionFull` over packed sequences: `bounds` holds one `[begin,
/// end)` u32 pair per row, the rows of that row's sequence (at most
/// `attention_full_max_rows` of them), and each row attends only there.
/// The device clamps the pairs; keeping them true is the caller's job.
pub fn attentionSegments(self: *Backend, queries: Buffer, keys: Buffer, values: Buffer, output: Buffer, bounds: Buffer, s: AttentionSegmentsShape) !void {
    if (s.heads == 0 or s.width == 0 or s.width > 256 or s.rows == 0 or s.rows > std.math.maxInt(u32) / s.heads) return error.InvalidShape;
    const row_floats = s.heads * s.width;
    if (s.q_stride < row_floats or s.kv_stride < row_floats or s.out_stride < row_floats) return error.InvalidShape;
    const q_len = ((s.rows - 1) * s.q_stride + row_floats) * 4;
    const kv_len = ((s.rows - 1) * s.kv_stride + row_floats) * 4;
    const out_len = ((s.rows - 1) * s.out_stride + row_floats) * 4;
    if (queries.len < q_len or keys.len < kv_len or values.len < kv_len or output.len < out_len or bounds.len < s.rows * 8 or bounds.offset % 8 != 0) return error.InvalidShape;
    if (Buffer.overlaps(output, out_len, queries, q_len) or Buffer.overlaps(output, out_len, keys, kv_len) or Buffer.overlaps(output, out_len, values, kv_len) or Buffer.overlaps(output, out_len, bounds, s.rows * 8)) return error.InvalidShape;
    const window: u32 = if (s.window) |w| @intCast(@min(w, std.math.maxInt(u32) - 1)) else std.math.maxInt(u32);
    const p: AttentionSegmentsParams = .{ .heads = @intCast(s.heads), .width = @intCast(s.width), .rows = @intCast(s.rows), .q_stride = @intCast(s.q_stride), .kv_stride = @intCast(s.kv_stride), .out_stride = @intCast(s.out_stride), .scale = s.scale, .window = window };
    try self.dispatch(.attention_segments, &.{ queries, keys, values, output, bounds }, p, @intCast(s.rows * s.heads), 256, .{});
}
pub const AttentionGemmParams = extern struct { query_heads: u32, kv_heads: u32, width: u32, rows: u32, q_stride: u32, kv_stride: u32, out_stride: u32, head_stride: u32, scale: f32, window: u32 };
pub const AttentionSegmentsGroupedShape = struct {
    query_heads: usize,
    /// Query head h reads kv head `h / (query_heads / kv_heads)`.
    kv_heads: usize,
    /// Query, key, and value head width: a multiple of 64, at most 512.
    width: usize,
    rows: usize,
    q_stride: usize,
    kv_stride: usize,
    out_stride: usize,
    scale: f32,
    /// Row i sees key j when |i − j| ≤ window; null sees its whole sequence.
    window: ?usize = null,
};
/// One query tile of `attentionSegmentsGrouped`: rows `[first, first +
/// rows)` (a multiple of 8, at most `attention_tile_rows`), the keys
/// `[key_lo, key_lo + span)` they may see (span a multiple of 32), and
/// their score strip of `rows · span` floats at `offset`.
pub const AttentionTile = extern struct { first: u32, rows: u32, key_lo: u32, span: u32, offset: u32 };
pub const attention_tile_rows = 32;
/// One dispatchable run of tiles: `tiles[first ..][0..count]`, whose
/// strips take `head_stride` floats per head, `max_span` the widest.
pub const AttentionChunk = struct { first: usize, count: usize, head_stride: usize, max_span: usize };

/// Appends the tiles of rows `[first, first + rows)` (one input with its
/// padding, `rows` a multiple of 8) to `tiles` at `count.*`, each with
/// the key range its rows see under `bounds` (host u32 pairs) and
/// `window`, as the kernels mask them; offsets are left 0 for `chunkTiles`.
pub fn attentionTiles(bounds: []const u32, total_rows: usize, window: ?usize, first: usize, rows: usize, tiles: []AttentionTile, count: *usize) !void {
    if (rows % 8 != 0 or first + rows > total_rows or bounds.len < 2 * total_rows) return error.InvalidShape;
    var at: usize = 0;
    while (at < rows) {
        const n = @min(rows - at, attention_tile_rows);
        var lo: usize = std.math.maxInt(usize);
        var hi: usize = 0;
        for (first + at..first + at + n) |r| {
            var a: usize = @min(bounds[2 * r], r);
            var b: usize = std.math.clamp(@as(usize, bounds[2 * r + 1]), r + 1, total_rows);
            if (window) |w| {
                a = @max(a, r -| w);
                b = @min(b, r + w + 1);
            }
            lo = @min(lo, a);
            hi = @max(hi, b);
        }
        if (count.* >= tiles.len) return error.InvalidShape;
        tiles[count.*] = .{ .first = @intCast(first + at), .rows = @intCast(n), .key_lo = @intCast(lo), .span = @intCast(std.mem.alignForward(usize, hi - lo, 32)), .offset = 0 };
        count.* += 1;
        at += n;
    }
}
/// Sets each tile's strip offset and splits `tiles` into chunks whose
/// strips fit `capacity` floats over `heads` heads; returns the chunk count.
pub fn chunkTiles(tiles: []AttentionTile, heads: usize, capacity: usize, chunks: []AttentionChunk) !usize {
    var count: usize = 0;
    var i: usize = 0;
    while (i < tiles.len) {
        if (count >= chunks.len) return error.InvalidShape;
        var chunk: AttentionChunk = .{ .first = i, .count = 0, .head_stride = 0, .max_span = 0 };
        while (i < tiles.len) : (i += 1) {
            const strip = @as(usize, tiles[i].rows) * tiles[i].span;
            if (chunk.count > 0 and heads * (chunk.head_stride + strip) > capacity) break;
            if (heads * strip > capacity) return error.InvalidShape;
            tiles[i].offset = @intCast(chunk.head_stride);
            chunk.head_stride += strip;
            chunk.max_span = @max(chunk.max_span, tiles[i].span);
            chunk.count += 1;
        }
        chunks[count] = chunk;
        count += 1;
    }
    return count;
}
/// `attentionSegments` with grouped-query heads, widths to 512, and no
/// bound on a sequence's length, over the query tiles of one chunk:
/// scores into `scores` (`query_heads · chunk.head_stride` floats),
/// their softmax, and P·V into `output`, F32 throughout. `tiles` is the
/// chunk's slice of the tile buffer (`attentionTiles`, `chunkTiles`),
/// checked here: ascending, disjoint, inside the rows. Queries and output
/// hold `attentionChunkRows(rows)` rows, keys and values `rows`, all
/// float4-aligned. A row's result depends on its tile's rows and keys
/// only. The device clamps the bounds; keeping them true is the caller's job.
pub fn attentionSegmentsGrouped(self: *Backend, queries: Buffer, keys: Buffer, values: Buffer, output: Buffer, bounds: Buffer, tiles: Buffer, chunk: AttentionChunk, scores: Buffer, s: AttentionSegmentsGroupedShape) !void {
    if (s.query_heads == 0 or s.kv_heads == 0 or s.query_heads % s.kv_heads != 0 or s.width == 0 or s.width % 64 != 0 or s.width > 512) return error.InvalidShape;
    if (s.rows == 0 or s.rows >= std.math.maxInt(u32) or !std.math.isFinite(s.scale)) return error.InvalidShape;
    const q_floats = s.query_heads * s.width;
    const kv_floats = s.kv_heads * s.width;
    if (s.q_stride < q_floats or s.q_stride % 4 != 0 or s.kv_stride < kv_floats or s.kv_stride % 4 != 0 or s.out_stride < q_floats or s.out_stride % 4 != 0) return error.InvalidShape;
    const padded = attentionChunkRows(s.rows);
    const q_len = ((padded - 1) * s.q_stride + q_floats) * 4;
    const kv_len = ((s.rows - 1) * s.kv_stride + kv_floats) * 4;
    const out_len = ((padded - 1) * s.out_stride + q_floats) * 4;
    if (queries.len < q_len or keys.len < kv_len or values.len < kv_len or output.len < out_len or bounds.len < s.rows * 8 or bounds.offset % 8 != 0) return error.InvalidShape;
    if (queries.offset % 16 != 0 or keys.offset % 16 != 0 or values.offset % 16 != 0 or output.offset % 16 != 0 or scores.offset % 16 != 0) return error.InvalidShape;
    const scores_len = s.query_heads * chunk.head_stride * 4;
    if (chunk.count == 0 or chunk.head_stride == 0 or chunk.head_stride >= std.math.maxInt(u32) or scores.len < scores_len) return error.InvalidShape;
    if (tiles.len < chunk.count * @sizeOf(AttentionTile) or tiles.offset % 4 != 0) return error.InvalidShape;
    if (Buffer.overlaps(output, out_len, queries, q_len) or Buffer.overlaps(output, out_len, keys, kv_len) or Buffer.overlaps(output, out_len, values, kv_len) or Buffer.overlaps(output, out_len, scores, scores_len)) return error.InvalidShape;
    // Overlapping tiles or strips would race, so the list is checked here.
    const list: []const AttentionTile = @as([*]const AttentionTile, @ptrCast(@alignCast(tiles.host)))[0..chunk.count];
    var next_row: usize = 0;
    var max_span: usize = 0;
    for (list) |t| {
        if (t.first < next_row or t.first % 8 != 0 or t.rows == 0 or t.rows % 8 != 0 or t.rows > attention_tile_rows or t.first + t.rows > padded) return error.InvalidShape;
        if (t.span == 0 or t.span % 32 != 0 or t.key_lo >= s.rows or t.offset % 32 != 0 or @as(usize, t.offset) + @as(usize, t.rows) * t.span > chunk.head_stride) return error.InvalidShape;
        next_row = t.first + t.rows;
        max_span = @max(max_span, t.span);
    }
    if (max_span != chunk.max_span) return error.InvalidShape;
    const window: u32 = if (s.window) |w| @intCast(@min(w, std.math.maxInt(u32) - 1)) else std.math.maxInt(u32);
    const p: AttentionGemmParams = .{ .query_heads = @intCast(s.query_heads), .kv_heads = @intCast(s.kv_heads), .width = @intCast(s.width), .rows = @intCast(s.rows), .q_stride = @intCast(s.q_stride), .kv_stride = @intCast(s.kv_stride), .out_stride = @intCast(s.out_stride), .head_stride = @intCast(chunk.head_stride), .scale = s.scale, .window = window };
    const n: u32 = @intCast(chunk.count);
    try self.dispatch2(.segments_scores, &.{ queries, keys, bounds, tiles, scores }, p, n, @intCast(s.query_heads * chunk.max_span / 32), 128);
    try self.dispatch2(.segments_softmax, &.{ tiles, scores }, p, n, @intCast(s.query_heads), 256);
    try self.dispatch2(.segments_values, &.{ scores, values, tiles, output }, p, n, @intCast(s.query_heads * s.width / 32), 128);
}
