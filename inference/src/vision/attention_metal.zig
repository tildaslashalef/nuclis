//! Bidirectional attention over a vision encoder's patch rows on Metal:
//! each row attends to every row of its window (one window for a whole-image
//! block). It runs the segments kernels with materialized scores
//! (`Backend.attentionSegmentsGrouped`), which reach several times the chunk
//! attention kernel's rate on these shapes (docs/engine/vision.md § Faster
//! encoders). A `Spans` holds one block kind's per-row bounds and query
//! tiles; the plan that owns it owns the score buffer the dispatches share.
const std = @import("std");
const metal = @import("../backends/metal/root.zig");

const Backend = metal.Backend;
const Buffer = metal.Buffer;

/// Floats of score strips one dispatch may use (128 MB); an image's
/// attention runs in as many chunks as its strips need.
pub const score_capacity = 32 << 20;

/// Rows `[begin, begin + count)` that attend among themselves (any
/// struct with these fields serves, such as Muse Glimmer's `Window`).
pub const Window = struct { begin: u32, count: u32 };

pub const Spans = struct {
    /// One u32 `[begin, end)` pair per row.
    bounds: Buffer,
    tiles: Buffer,
    /// Host: the chunks of the last `plan`.
    chunks: []Backend.AttentionChunk,
    chunk_count: usize,
    rows: usize,
    heads: usize,

    /// Buffers for up to `max_rows` rows; `created` counts their bytes.
    pub fn init(alloc: std.mem.Allocator, backend: *Backend, max_rows: usize, created: *usize) !Spans {
        const max_tiles = Backend.attentionChunkRows(max_rows) / 8;
        const chunks = try alloc.alloc(Backend.AttentionChunk, max_tiles);
        errdefer alloc.free(chunks);
        const bounds = try backend.create(max_rows * 8);
        const tiles = try backend.create(max_tiles * @sizeOf(Backend.AttentionTile));
        created.* += max_rows * 8 + max_tiles * @sizeOf(Backend.AttentionTile);
        return .{ .bounds = bounds, .tiles = tiles, .chunks = chunks, .chunk_count = 0, .rows = 0, .heads = 0 };
    }
    /// Device buffers belong to the backend and go with it.
    pub fn deinit(self: *Spans, alloc: std.mem.Allocator) void {
        alloc.free(self.chunks);
        self.* = undefined;
    }

    /// Lays out `rows` rows for `heads` query heads, each row seeing its
    /// window's rows; `windows` must cover `[0, rows)` in order. Writes the
    /// host-visible bounds and tiles, so call it before the command buffer
    /// that attends, and not while one that reads them is in flight.
    pub fn plan(self: *Spans, windows: anytype, rows: usize, heads: usize) !void {
        if (rows == 0 or 2 * rows * 4 > self.bounds.len or heads == 0) return error.InvalidShape;
        const pairs: []u32 = @as([*]u32, @ptrCast(@alignCast(self.bounds.host)))[0 .. 2 * rows];
        var next: usize = 0;
        for (windows) |w| {
            if (w.begin != next or w.count == 0 or w.begin + w.count > rows) return error.InvalidShape;
            for (w.begin..w.begin + w.count) |r| {
                pairs[2 * r] = w.begin;
                pairs[2 * r + 1] = w.begin + w.count;
            }
            next = w.begin + w.count;
        }
        if (next != rows) return error.InvalidShape;
        const list: []Backend.AttentionTile = @as([*]Backend.AttentionTile, @ptrCast(@alignCast(self.tiles.host)))[0..self.chunks.len];
        var count: usize = 0;
        try Backend.attentionTiles(pairs, rows, null, 0, Backend.attentionChunkRows(rows), list, &count);
        self.chunk_count = try Backend.chunkTiles(list[0..count], heads, score_capacity, self.chunks);
        self.rows = rows;
        self.heads = heads;
    }

    pub const Shape = struct { width: usize, q_stride: usize, kv_stride: usize, out_stride: usize, scale: f32 };
    /// Records the attention of the planned rows: queries and output hold
    /// `attentionChunkRows(rows)` rows (the padding rows come out zero),
    /// keys and values `rows`. `scores` holds `score_capacity` floats.
    pub fn attend(self: *const Spans, b: *Backend, queries: Buffer, keys: Buffer, values: Buffer, output: Buffer, scores: Buffer, s: Shape) !void {
        for (self.chunks[0..self.chunk_count]) |chunk| {
            const tiles = self.tiles.slice(chunk.first * @sizeOf(Backend.AttentionTile), chunk.count * @sizeOf(Backend.AttentionTile));
            try b.attentionSegmentsGrouped(queries, keys, values, output, self.bounds, tiles, chunk, scores, .{
                .query_heads = self.heads,
                .kv_heads = self.heads,
                .width = s.width,
                .rows = self.rows,
                .q_stride = s.q_stride,
                .kv_stride = s.kv_stride,
                .out_stride = s.out_stride,
                .scale = s.scale,
            });
        }
    }
};
