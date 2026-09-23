//! The Muse Glimmer projector (`clip.projector_type = muse-glimmer`): a
//! 50-block pre-LayerNorm ViT over 14-pixel patches whose attention runs
//! inside 32×32-patch windows except on every fourth and the last block,
//! then a 2×2 pixel shuffle into a three-layer MLP at the language model's
//! width. Rows run in window order (`Layout`) from the patch embedding to
//! the post-norm; the rotary positions come from each patch's grid cell,
//! never from its row. `bind` validates the companion file; `Runtime` is the
//! CPU reference (F64 accumulation); the host tables are shared with the
//! Metal plan. Facts and provenance: docs/reference/vision.md § Muse Glimmer.
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const weights = @import("../runtime/weights.zig");
const cpu = @import("../backends/cpu/root.zig");
const quant = @import("../quant/decode.zig");
const preprocess = @import("preprocess.zig");
const inventory = @import("../models/inventory.zig");
const qwen3vl = @import("qwen3vl.zig");

const Tensor = gguf.Tensor;
/// The Metal plan, the same schedule on the device.
pub const Plan = @import("muse_glimmer_metal.zig").Plan;

pub const projector_type = "muse-glimmer";
pub const hidden = 1536;
pub const ffn = 8960;
pub const heads = 16;
pub const head_dim = 96;
pub const blocks = 50;
pub const patch = 14;
pub const merge = 2;
pub const output_width = 6656;
pub const adapter_width = 4096;
pub const merged_width = hidden * merge * merge;
pub const patch_values = patch * patch * 3;
/// Pixels per token side: one merge block of patches.
pub const token_side = patch * merge;
/// The learned position table is a `position_side²` grid; the attention
/// windows are `window × window` patches, the table's side.
pub const position_side = 32;
pub const window = position_side;
/// Blocks `3, 7, …, 47` and the last attend globally, the rest per window.
pub const sparse_factor = 4;
pub const norm_epsilon: f32 = 1e-5;
pub const rope_base: f64 = 10000;
pub const rope_pairs = head_dim / 2;
/// The reference's token bounds (`set_limit_image_tokens(1, 4096)`).
pub const min_tokens = 1;
pub const max_tokens = 4096;
pub const max_patches = max_tokens * merge * merge;

pub fn isGlobal(layer: usize) bool {
    return layer == blocks - 1 or (layer + 1) % sparse_factor == 0;
}

pub const Error = qwen3vl.Error;
pub const Norm = qwen3vl.Norm;
pub const Linear = qwen3vl.Linear;
pub const Block = struct { norm1: Norm, query: Linear, key: Linear, value: Linear, output: Linear, norm2: Norm, up: Linear, down: Linear };

pub const Binding = struct {
    /// `[hidden, 3·14·14]` F32, columns `c · 196 + ky · 14 + kx`; no bias.
    patch_embedding: *const Tensor,
    /// `[position_side², hidden]` F32.
    position_embedding: *const Tensor,
    pre_norm: Norm,
    layers: [blocks]Block,
    post_norm: Norm,
    /// `6144 → 4096 → 4096 → 6656`, no biases.
    adapter: [3]*const Tensor,
    mean: [3]f32,
    std: [3]f32,
    tensors: u32,
    bytes: u64,
};

/// Whether `doc` is a Muse Glimmer projector.
pub fn matches(doc: *const gguf.Document) bool {
    const kind = qwen3vl.stringValue(doc, "clip.projector_type") catch return false;
    return std.mem.eql(u8, kind, projector_type);
}

fn validateMetadata(doc: *const gguf.Document) Error!void {
    if (!std.mem.eql(u8, try qwen3vl.stringValue(doc, "general.architecture"), "clip")) return error.UnsupportedArchitecture;
    if (!matches(doc)) return error.UnsupportedProjector;
    const settings = [_]struct { key: []const u8, value: u64 }{
        .{ .key = "clip.vision.projection_dim", .value = output_width },
        .{ .key = "clip.vision.patch_size", .value = patch },
        .{ .key = "clip.vision.embedding_length", .value = hidden },
        .{ .key = "clip.vision.feed_forward_length", .value = ffn },
        .{ .key = "clip.vision.block_count", .value = blocks },
        .{ .key = "clip.vision.attention.head_count", .value = heads },
    };
    for (settings) |s| if (try qwen3vl.unsignedValue(doc, s.key) != s.value) return error.UnsupportedConfiguration;
    // The reference defaults the merge to 2 when the key is absent.
    if (doc.get("clip.vision.spatial_merge_size") != null and try qwen3vl.unsignedValue(doc, "clip.vision.spatial_merge_size") != merge) return error.UnsupportedConfiguration;
    switch (doc.get("clip.vision.attention.layer_norm_epsilon") orelse return error.MissingMetadata) {
        .float => |f| if (@abs(f - norm_epsilon) > 1e-9) return error.UnsupportedConfiguration,
        else => return error.InvalidMetadata,
    }
}

/// Encodings each tensor class may have: the file's own (the block
/// matrices are a K-quant mix, the adapter BF16); anything else is refused.
const block_matrix = [_]u32{ 12, 14 }; // Q4_K, Q6_K
const adapter_matrix = [_]u32{ 30, 0 }; // BF16, F32
const vector_only = [_]u32{0};

const Binder = struct {
    remaining: std.StringHashMap(*const Tensor),
    tensors: u32 = 0,
    bytes: u64 = 0,

    fn take(self: *Binder, name: []const u8, dimensions: []const u64, encodings: []const u32) Error!*const Tensor {
        const entry = self.remaining.fetchRemove(name) orelse return error.MissingTensor;
        const tensor = entry.value;
        if (!std.mem.eql(u64, tensor.dimensions, dimensions)) return error.InvalidTensorShape;
        if (std.mem.indexOfScalar(u32, encodings, tensor.encoding_id) == null) return error.UnsupportedTensorEncoding;
        self.tensors += 1;
        self.bytes = std.math.add(u64, self.bytes, tensor.bytes) catch return error.InvalidTensorShape;
        return tensor;
    }
    fn named(self: *Binder, comptime fmt: []const u8, args: anytype, dimensions: []const u64, encodings: []const u32) Error!*const Tensor {
        var name: [96]u8 = undefined;
        return self.take(std.fmt.bufPrint(&name, fmt, args) catch unreachable, dimensions, encodings);
    }
    fn norm(self: *Binder, comptime prefix: []const u8, args: anytype) Error!Norm {
        return .{ .weight = try self.named(prefix ++ ".weight", args, &.{hidden}, &vector_only), .bias = try self.named(prefix ++ ".bias", args, &.{hidden}, &vector_only) };
    }
    fn linear(self: *Binder, comptime prefix: []const u8, args: anytype, columns: u64, rows: u64) Error!Linear {
        return .{ .weight = try self.named(prefix ++ ".weight", args, &.{ columns, rows }, &block_matrix), .bias = try self.named(prefix ++ ".bias", args, &.{rows}, &vector_only) };
    }
};

/// doc must come from successful GGUF parsing; the returned tensor references
/// borrow it. Allocations are lookup storage freed before return.
pub fn bind(alloc: std.mem.Allocator, doc: *const gguf.Document) Error!Binding {
    try validateMetadata(doc);
    var binder: Binder = .{ .remaining = .init(alloc) };
    defer binder.remaining.deinit();
    for (doc.tensors) |*tensor| {
        const slot = try binder.remaining.getOrPut(tensor.name);
        if (slot.found_existing) return error.DuplicateTensor;
        slot.value_ptr.* = tensor;
    }
    var result: Binding = undefined;
    result.mean = try qwen3vl.floatTriple(doc, "clip.vision.image_mean");
    result.std = try qwen3vl.floatTriple(doc, "clip.vision.image_std");
    for (result.std) |s| if (!(s > 0)) return error.UnsupportedConfiguration;
    result.patch_embedding = try binder.take("v.patch_embd.weight", &.{ patch, patch, 3, hidden }, &vector_only);
    result.position_embedding = try binder.take("v.position_embd.weight", &.{ hidden, position_side * position_side }, &vector_only);
    result.pre_norm = try binder.norm("v.pre_ln", .{});
    for (&result.layers, 0..) |*layer, i| layer.* = .{
        .norm1 = try binder.norm("v.blk.{d}.ln1", .{i}),
        .query = try binder.linear("v.blk.{d}.attn_q", .{i}, hidden, hidden),
        .key = try binder.linear("v.blk.{d}.attn_k", .{i}, hidden, hidden),
        .value = try binder.linear("v.blk.{d}.attn_v", .{i}, hidden, hidden),
        .output = try binder.linear("v.blk.{d}.attn_out", .{i}, hidden, hidden),
        .norm2 = try binder.norm("v.blk.{d}.ln2", .{i}),
        .up = try binder.linear("v.blk.{d}.ffn_up", .{i}, hidden, ffn),
        .down = try binder.linear("v.blk.{d}.ffn_down", .{i}, ffn, hidden),
    };
    result.post_norm = try binder.norm("v.post_ln", .{});
    result.adapter = .{
        try binder.take("mm.0.weight", &.{ merged_width, adapter_width }, &adapter_matrix),
        try binder.take("mm.1.weight", &.{ adapter_width, adapter_width }, &adapter_matrix),
        try binder.take("mm.2.weight", &.{ adapter_width, output_width }, &adapter_matrix),
    };
    if (binder.remaining.count() != 0) return error.UnexpectedTensor;
    result.tensors = binder.tensors;
    result.bytes = binder.bytes;
    return result;
}

/// The patch grid of a resized image; both sides are even.
pub const Grid = struct {
    width_patches: u32,
    height_patches: u32,
    pub fn patches(self: Grid) usize {
        return @as(usize, self.width_patches) * self.height_patches;
    }
    pub fn widthTokens(self: Grid) u32 {
        return self.width_patches / merge;
    }
    pub fn heightTokens(self: Grid) u32 {
        return self.height_patches / merge;
    }
    pub fn tokens(self: Grid) usize {
        return self.patches() / (merge * merge);
    }
};

/// The reference's grid (transformers' `get_aspect_ratio_preserving_size`
/// on 28-pixel tokens): scale the token counts down to `max` keeping the
/// aspect ratio, then of the floor/ceil pairs under the cap take the one
/// whose `h/w` is closest to the image's (ties to more tokens). The image is
/// then stretched to the grid, not letterboxed. When nothing fits (an image
/// thinner than one token per `max`), the rounded pair is clamped under the
/// cap, where the reference would exceed it.
pub fn gridFor(size: preprocess.Size, max: u32) Grid {
    const unit: f64 = token_side;
    var nph = @as(f64, @floatFromInt(size.height)) / unit;
    var npw = @as(f64, @floatFromInt(size.width)) / unit;
    const ratio = if (nph > 0) npw / nph else 1.0;
    const cap: f64 = @floatFromInt(max);
    if (nph * npw > cap) {
        nph = @sqrt(cap / ratio);
        npw = nph * ratio;
    }
    const hs = [2]i64{ @intFromFloat(@floor(nph)), @intFromFloat(@ceil(nph)) };
    const ws = [2]i64{ @intFromFloat(@floor(npw)), @intFromFloat(@ceil(npw)) };
    const target = @as(f64, @floatFromInt(size.height)) / @as(f64, @floatFromInt(size.width));
    var best_h: i64 = -1;
    var best_w: i64 = -1;
    var best_d: f64 = 0;
    for (hs) |h| for (ws) |w| {
        if (h < 1 or w < 1 or h * w > max) continue;
        const d = @abs(@as(f64, @floatFromInt(h)) / @as(f64, @floatFromInt(w)) - target);
        if (best_h < 0 or d < best_d or (d == best_d and h * w > best_h * best_w)) {
            best_h = h;
            best_w = w;
            best_d = d;
        }
    };
    if (best_h < 0) {
        best_h = @max(1, @as(i64, @intFromFloat(@round(nph))));
        best_w = @max(1, @as(i64, @intFromFloat(@round(npw))));
        if (best_h * best_w > max) {
            if (best_h > best_w) best_h = @max(1, @divFloor(max, best_w)) else best_w = @max(1, @divFloor(max, best_h));
        }
    }
    return .{ .width_patches = @intCast(best_w * merge), .height_patches = @intCast(best_h * merge) };
}

/// One attention window: rows `[begin, begin + count)` in window order.
pub const Window = struct { begin: u32, count: u32 };

/// The row order the blocks run in: `window × window`-patch windows in
/// raster order, patches raster inside each (edge windows partial), and
/// the windows' row ranges. `order[r]` is the raster index of row r's
/// patch; `row_of` its inverse.
pub const Layout = struct {
    order: []u32,
    row_of: []u32,
    windows: []Window,

    pub fn init(alloc: std.mem.Allocator, grid: Grid) !Layout {
        const n = grid.patches();
        const gw = grid.width_patches;
        const gh = grid.height_patches;
        const across = (gw + window - 1) / window;
        const down = (gh + window - 1) / window;
        const order = try alloc.alloc(u32, n);
        errdefer alloc.free(order);
        const row_of = try alloc.alloc(u32, n);
        errdefer alloc.free(row_of);
        const windows = try alloc.alloc(Window, across * down);
        errdefer alloc.free(windows);
        var r: u32 = 0;
        for (0..down) |wy| for (0..across) |wx| {
            const begin = r;
            const y0 = wy * window;
            const x0 = wx * window;
            for (y0..@min(y0 + window, gh)) |y| for (x0..@min(x0 + window, gw)) |x| {
                const p: u32 = @intCast(y * gw + x);
                order[r] = p;
                row_of[p] = r;
                r += 1;
            };
            windows[wy * across + wx] = .{ .begin = begin, .count = r - begin };
        };
        return .{ .order = order, .row_of = row_of, .windows = windows };
    }
    pub fn deinit(self: *Layout, alloc: std.mem.Allocator) void {
        alloc.free(self.order);
        alloc.free(self.row_of);
        alloc.free(self.windows);
        self.* = undefined;
    }
};

/// The rotary table in window order: per row `rope_pairs` (cos, sin)
/// pairs for the adjacent-pair rotation. Pair j < 24 turns channels
/// (2j, 2j + 1) by the patch's column + 1 at `base^(−j/24)`; pair j ≥ 24
/// turns (2j, 2j + 1) by its row + 1 at `base^(−(j−24)/24)` — the
/// reference's two half-width normal ropes with 1-indexed positions. F64
/// angles rounded to F32.
pub fn ropeTable(out: []f32, grid: Grid, layout: Layout) !void {
    const n = grid.patches();
    if (out.len < n * rope_pairs * 2 or layout.order.len != n) return error.InvalidShape;
    const half = rope_pairs / 2;
    for (0..n) |r| {
        const p = layout.order[r];
        const column: f64 = @floatFromInt(p % grid.width_patches + 1);
        const row: f64 = @floatFromInt(p / grid.width_patches + 1);
        for (0..rope_pairs) |j| {
            const position = if (j < half) column else row;
            const exponent = -@as(f64, @floatFromInt(j % half)) / @as(f64, @floatFromInt(half));
            const theta = position * std.math.pow(f64, rope_base, exponent);
            out[(r * rope_pairs + j) * 2] = @floatCast(@cos(theta));
            out[(r * rope_pairs + j) * 2 + 1] = @floatCast(@sin(theta));
        }
    }
}

/// The learned position rows in window order: the 32×32 table resized to
/// the patch grid bilinearly with half-pixel centres and clamped edges (the
/// reference's `ggml_interpolate`, no aligned corners), in F32 as it
/// computes it; the table itself when the grid is 32×32. `out` holds
/// `grid.patches() · hidden` values; `scratch` four rows of `hidden`.
pub fn positionRows(view: weights.View, table: *const Tensor, grid: Grid, layout: Layout, out: []f32, scratch: []f32) !void {
    const n = grid.patches();
    if (out.len < n * hidden or scratch.len < 4 * hidden or layout.order.len != n) return error.InvalidShape;
    const side = position_side;
    for (0..n) |r| {
        const p = layout.order[r];
        const x = p % grid.width_patches;
        const y = p / grid.width_patches;
        const row = out[r * hidden ..][0..hidden];
        if (grid.width_patches == side and grid.height_patches == side) {
            try view.row(table, @as(usize, y) * side + x, row);
            continue;
        }
        const sx = sample(x, grid.width_patches);
        const sy = sample(y, grid.height_patches);
        const corners = [4][2]usize{ .{ sx.first, sy.first }, .{ sx.second, sy.first }, .{ sx.first, sy.second }, .{ sx.second, sy.second } };
        for (corners, 0..) |c, k| try view.row(table, c[1] * side + c[0], scratch[k * hidden ..][0..hidden]);
        const a = scratch[0..hidden];
        const b = scratch[hidden..][0..hidden];
        const c = scratch[2 * hidden ..][0..hidden];
        const d = scratch[3 * hidden ..][0..hidden];
        for (row, 0..) |*o, i| o.* = a[i] * (1 - sx.frac) * (1 - sy.frac) + b[i] * sx.frac * (1 - sy.frac) + c[i] * (1 - sx.frac) * sy.frac + d[i] * sx.frac * sy.frac;
    }
}
/// One axis of the half-pixel resize from `position_side` to `count`.
fn sample(index: u32, count: u32) struct { first: usize, second: usize, frac: f32 } {
    const scale: f32 = @as(f32, @floatFromInt(count)) / @as(f32, position_side);
    const v = (@as(f32, @floatFromInt(index)) + 0.5) / scale - 0.5;
    const floor: i64 = @intFromFloat(@floor(v));
    const first: i64 = std.math.clamp(floor, 0, position_side - 1);
    const second: i64 = std.math.clamp(floor + 1, 0, position_side - 1);
    const frac = std.math.clamp(v - @as(f32, @floatFromInt(first)), 0, 1);
    return .{ .first = @intCast(first), .second = @intCast(second), .frac = frac };
}

/// The patch kernel as an F32 `[hidden × columns]` matrix, the file's
/// `c · 196 + ky · 14 + kx` columns (the layout `preprocess.patches`
/// writes) zero-padded from 588 to `columns`. Owned by the caller.
pub fn patchKernel(alloc: std.mem.Allocator, view: weights.View, binding: *const Binding, columns: usize) ![]f32 {
    if (columns < patch_values) return error.InvalidShape;
    const raw = try view.bytes(binding.patch_embedding);
    if (raw.len != hidden * patch_values * 4) return error.InvalidShape;
    const out = try alloc.alloc(f32, hidden * columns);
    @memset(out, 0);
    // GGUF dimensions [kx, ky, c, o]: kx fastest, so row o is contiguous.
    for (0..hidden) |o| for (0..patch_values) |i| {
        out[o * columns + i] = @bitCast(std.mem.readInt(u32, raw[(o * patch_values + i) * 4 ..][0..4], .little));
    };
    return out;
}

/// The patch rows the kernel reads: `preprocess.patches` with merge 1
/// (raster), normalized by the file's mean/std and rounded to F16 (the
/// reference's im2col is F16).
pub fn patchOptions(mean: [3]f32, std_dev: [3]f32) preprocess.PatchOptions {
    return .{ .patch = patch, .merge = 1, .mean = mean, .std = std_dev, .half = true };
}

/// The pixel shuffle: merged token o of the `grid.widthTokens()`-wide token
/// grid holds, at element `c · 4 + s`, channel c of its patch s =
/// `2·ry + rx` (patch `(2·ox + rx, 2·oy + ry)`), read from `rows` in window
/// order. `out` holds `grid.tokens() · merged_width` values.
pub fn shuffle(rows: []const f32, grid: Grid, layout: Layout, out: []f32) !void {
    const tokens = grid.tokens();
    if (rows.len < grid.patches() * hidden or out.len < tokens * merged_width) return error.InvalidShape;
    const wt = grid.widthTokens();
    for (0..tokens) |o| {
        const oy = o / wt;
        const ox = o % wt;
        const dst = out[o * merged_width ..][0..merged_width];
        for (0..merge * merge) |s| {
            const p = (oy * merge + s / merge) * grid.width_patches + ox * merge + s % merge;
            const src = rows[@as(usize, layout.row_of[p]) * hidden ..][0..hidden];
            for (src, 0..) |v, c| dst[c * merge * merge + s] = v;
        }
    }
}

/// `out[t] = W · input[t] (+ bias)` for `n` rows, decoding each weight row
/// once; F64 accumulation as `cpu.matvec`. `scratch` holds one decoded row.
fn rowsTimes(matrix: cpu.Matrix, input: []const f32, n: usize, output: []f32, bias: ?[]const f32, scratch: []f32) !void {
    const row_bytes = matrix.bytes.len / matrix.rows;
    try quant.validateRow(matrix.encoding, row_bytes, matrix.columns);
    const decoded = scratch[0..matrix.columns];
    for (0..matrix.rows) |r| {
        try quant.row(matrix.encoding, matrix.bytes[r * row_bytes ..][0..row_bytes], decoded);
        const b: f64 = if (bias) |v| v[r] else 0;
        for (0..n) |t| {
            var sum: f64 = 0;
            for (decoded, input[t * matrix.columns ..][0..matrix.columns]) |w, x| sum += @as(f64, w) * x;
            output[t * matrix.rows + r] = @floatCast(sum + b);
        }
    }
}

/// The CPU reference: matrix products with F64 accumulation, LayerNorm and
/// attention in F64, the exact (erf) GELU.
pub const Runtime = struct {
    alloc: std.mem.Allocator,
    view: weights.View,
    binding: *const Binding,
    kernel: []f32,
    vectors: Vectors,

    const Pair = struct { weight: []f32, bias: []f32 };
    const BlockVectors = struct { n1: Pair, qb: []f32, kb: []f32, vb: []f32, ob: []f32, n2: Pair, ub: []f32, db: []f32 };
    const Vectors = struct { pre: Pair, layers: [blocks]BlockVectors, post: Pair };

    pub fn init(alloc: std.mem.Allocator, view: weights.View, binding: *const Binding) !Runtime {
        var loaded: std.ArrayList([]f32) = .empty;
        defer loaded.deinit(alloc);
        errdefer for (loaded.items) |v| alloc.free(v);
        const L = struct {
            list: *std.ArrayList([]f32),
            a: std.mem.Allocator,
            v: weights.View,
            fn one(self: @This(), t: *const Tensor) ![]f32 {
                const out = try self.v.vector(self.a, t);
                errdefer self.a.free(out);
                try self.list.append(self.a, out);
                return out;
            }
            fn pair(self: @This(), n: Norm) !Pair {
                return .{ .weight = try self.one(n.weight), .bias = try self.one(n.bias) };
            }
        };
        const l: L = .{ .list = &loaded, .a = alloc, .v = view };
        var vectors: Vectors = undefined;
        vectors.pre = try l.pair(binding.pre_norm);
        for (&vectors.layers, binding.layers) |*out, layer| out.* = .{
            .n1 = try l.pair(layer.norm1),
            .qb = try l.one(layer.query.bias),
            .kb = try l.one(layer.key.bias),
            .vb = try l.one(layer.value.bias),
            .ob = try l.one(layer.output.bias),
            .n2 = try l.pair(layer.norm2),
            .ub = try l.one(layer.up.bias),
            .db = try l.one(layer.down.bias),
        };
        vectors.post = try l.pair(binding.post_norm);
        const kernel = try patchKernel(alloc, view, binding, patch_values);
        loaded.clearRetainingCapacity();
        return .{ .alloc = alloc, .view = view, .binding = binding, .kernel = kernel, .vectors = vectors };
    }
    pub fn deinit(self: *Runtime) void {
        const a = self.alloc;
        a.free(self.kernel);
        const v = self.vectors;
        for ([_][]f32{ v.pre.weight, v.pre.bias, v.post.weight, v.post.bias }) |x| a.free(x);
        for (v.layers) |b| for ([_][]f32{ b.n1.weight, b.n1.bias, b.qb, b.kb, b.vb, b.ob, b.n2.weight, b.n2.bias, b.ub, b.db }) |x| a.free(x);
        self.* = undefined;
    }

    /// Encodes `patches` (raster rows from `patchOptions`) of an image of
    /// `grid` into `out`: `grid.tokens() × output_width` feature rows.
    pub fn encode(self: *Runtime, patches: preprocess.Patches, grid: Grid, out: []f32) !void {
        if (out.len < grid.tokens() * output_width) return error.InvalidShape;
        return self.run(patches, grid, blocks, .{ .features = out });
    }
    /// The residual rows (window order, `grid.patches() × hidden`) after the
    /// first `count` blocks, for the checks' traces.
    pub fn encodeResidual(self: *Runtime, patches: preprocess.Patches, grid: Grid, count: usize, rows: []f32) !void {
        if (count > blocks or rows.len < grid.patches() * hidden) return error.InvalidShape;
        return self.run(patches, grid, count, .{ .residual = rows });
    }

    fn run(self: *Runtime, patches: preprocess.Patches, grid: Grid, block_count: usize, result: Result) !void {
        const n = grid.patches();
        if (n == 0 or n > max_patches or patches.row != patch_values or patches.count() != n or patches.width_patches != grid.width_patches) return error.InvalidShape;
        const tokens = grid.tokens();
        const a = self.alloc;
        var layout = try Layout.init(a, grid);
        defer layout.deinit(a);
        const x = try a.alloc(f32, n * hidden);
        defer a.free(x);
        const h = try a.alloc(f32, n * hidden);
        defer a.free(h);
        const q = try a.alloc(f32, n * hidden);
        defer a.free(q);
        const k = try a.alloc(f32, n * hidden);
        defer a.free(k);
        const v = try a.alloc(f32, n * hidden);
        defer a.free(v);
        const attn = try a.alloc(f32, n * hidden);
        defer a.free(attn);
        const up = try a.alloc(f32, @max(n * ffn, n * patch_values, tokens * (merged_width + 2 * adapter_width)));
        defer a.free(up);
        const rope = try a.alloc(f32, n * rope_pairs * 2);
        defer a.free(rope);
        const scratch = try a.alloc(f32, @max(ffn, merged_width) + 4 * hidden);
        defer a.free(scratch);
        const scores = try a.alloc(f64, n);
        defer a.free(scores);
        try ropeTable(rope, grid, layout);

        // The patch rows gathered into window order, embedded, plus positions.
        const gathered = up[0 .. n * patch_values];
        for (0..n) |r| @memcpy(gathered[r * patch_values ..][0..patch_values], patches.values[@as(usize, layout.order[r]) * patch_values ..][0..patch_values]);
        const kernel: cpu.Matrix = .{ .encoding = 0, .rows = hidden, .columns = patch_values, .bytes = std.mem.sliceAsBytes(self.kernel) };
        try rowsTimes(kernel, gathered, n, x, null, scratch);
        try positionRows(self.view, self.binding.position_embedding, grid, layout, h, scratch);
        for (x, h) |*e, p| e.* += p;
        const vec = self.vectors;
        for (0..n) |t| qwen3vl.layerNormEpsilon(x[t * hidden ..][0..hidden], x[t * hidden ..][0..hidden], vec.pre.weight, vec.pre.bias, norm_epsilon);

        for (self.binding.layers[0..block_count], vec.layers[0..block_count], 0..) |layer, lv, il| {
            for (0..n) |t| qwen3vl.layerNormEpsilon(x[t * hidden ..][0..hidden], h[t * hidden ..][0..hidden], lv.n1.weight, lv.n1.bias, norm_epsilon);
            try rowsTimes(try self.view.matrix(layer.query.weight), h, n, q, lv.qb, scratch);
            try rowsTimes(try self.view.matrix(layer.key.weight), h, n, k, lv.kb, scratch);
            try rowsTimes(try self.view.matrix(layer.value.weight), h, n, v, lv.vb, scratch);
            for (0..n) |r| {
                const table = rope[r * rope_pairs * 2 ..][0 .. rope_pairs * 2];
                for (0..heads) |head| {
                    rotate(q[r * hidden + head * head_dim ..][0..head_dim], table);
                    rotate(k[r * hidden + head * head_dim ..][0..head_dim], table);
                }
            }
            if (isGlobal(il))
                attention(q, k, v, attn, .{ .begin = 0, .count = @intCast(n) }, scores)
            else for (layout.windows) |w| attention(q, k, v, attn, w, scores);
            try rowsTimes(try self.view.matrix(layer.output.weight), attn, n, h, lv.ob, scratch);
            for (x, h) |*e, r| e.* += r;
            for (0..n) |t| qwen3vl.layerNormEpsilon(x[t * hidden ..][0..hidden], h[t * hidden ..][0..hidden], lv.n2.weight, lv.n2.bias, norm_epsilon);
            const u = up[0 .. n * ffn];
            try rowsTimes(try self.view.matrix(layer.up.weight), h, n, u, lv.ub, scratch);
            for (u) |*e| e.* = cpu.geluErf(e.*);
            try rowsTimes(try self.view.matrix(layer.down.weight), u, n, h, lv.db, scratch);
            for (x, h) |*e, r| e.* += r;
        }
        const out = switch (result) {
            .residual => |rows| {
                @memcpy(rows[0 .. n * hidden], x);
                return;
            },
            .features => |f| f,
        };
        for (0..n) |t| qwen3vl.layerNormEpsilon(x[t * hidden ..][0..hidden], h[t * hidden ..][0..hidden], vec.post.weight, vec.post.bias, norm_epsilon);

        const merged = up[0 .. tokens * merged_width];
        try shuffle(h, grid, layout, merged);
        const mid = up[tokens * merged_width ..][0 .. tokens * adapter_width];
        const mid2 = up[tokens * (merged_width + adapter_width) ..][0 .. tokens * adapter_width];
        try rowsTimes(try self.view.matrix(self.binding.adapter[0]), merged, tokens, mid, null, scratch);
        for (mid) |*e| e.* = cpu.geluErf(e.*);
        try rowsTimes(try self.view.matrix(self.binding.adapter[1]), mid, tokens, mid2, null, scratch);
        for (mid2) |*e| e.* = cpu.geluErf(e.*);
        try rowsTimes(try self.view.matrix(self.binding.adapter[2]), mid2, tokens, out[0 .. tokens * output_width], null, scratch);
    }
};

/// What an encode leaves: the feature rows, or a check's residual rows.
pub const Result = union(enum) { features: []f32, residual: []f32 };

/// Rotates one head's channels in adjacent pairs `(2j, 2j + 1)` by the
/// table's pair j.
fn rotate(head: []f32, table: []const f32) void {
    for (0..rope_pairs) |j| {
        const c: f64 = table[j * 2];
        const s: f64 = table[j * 2 + 1];
        const a: f64 = head[2 * j];
        const b: f64 = head[2 * j + 1];
        head[2 * j] = @floatCast(a * c - b * s);
        head[2 * j + 1] = @floatCast(a * s + b * c);
    }
}

/// Bidirectional attention among the rows of `w` (queries and keys alike),
/// scale 1/√96; `out` rows of `w` get `hidden` values.
fn attention(q: []const f32, k: []const f32, v: []const f32, out: []f32, w: Window, scores: []f64) void {
    const scale = 1.0 / @sqrt(@as(f64, head_dim));
    const begin: usize = w.begin;
    const end: usize = begin + w.count;
    for (0..heads) |head| {
        const off = head * head_dim;
        for (begin..end) |i| {
            const query = q[i * hidden + off ..][0..head_dim];
            var max: f64 = -std.math.inf(f64);
            for (begin..end) |j| {
                const key = k[j * hidden + off ..][0..head_dim];
                var dot: f64 = 0;
                for (query, key) |a, b| dot += @as(f64, a) * b;
                scores[j] = dot * scale;
                max = @max(max, scores[j]);
            }
            var denominator: f64 = 0;
            for (scores[begin..end]) |*s| {
                s.* = @exp(s.* - max);
                denominator += s.*;
            }
            const o = out[i * hidden + off ..][0..head_dim];
            for (o, 0..) |*e, d| {
                var sum: f64 = 0;
                for (begin..end) |j| sum += scores[j] * v[j * hidden + off + d];
                e.* = @floatCast(sum / denominator);
            }
        }
    }
}

pub fn inventoryDocument(gpa: std.mem.Allocator) !gguf.Document {
    return inventory.document(gpa, @embedFile("fixtures/muse-glimmer-mmproj.json"));
}

test "bind accepts the pinned projector inventory and refuses another encoding" {
    const alloc = std.testing.allocator;
    var doc = try inventoryDocument(alloc);
    defer doc.storage.deinit();
    const binding = try bind(alloc, &doc);
    try std.testing.expectEqual(@as(u32, 809), binding.tensors);
    try std.testing.expectEqual([3]f32{ 0.5, 0.5, 0.5 }, binding.mean);
    try std.testing.expectEqual(@as(u32, 14), binding.layers[0].value.weight.encoding_id);
    const tensors = try alloc.dupe(Tensor, doc.tensors);
    defer alloc.free(tensors);
    for (tensors) |*t| if (std.mem.eql(u8, t.name, "v.blk.7.ffn_up.weight")) {
        t.encoding_id = 8; // Q8_0
    };
    var other = doc;
    other.tensors = tensors;
    try std.testing.expectError(error.UnsupportedTensorEncoding, bind(alloc, &other));
    var fewer = doc;
    fewer.tensors = doc.tensors[1..];
    try std.testing.expectError(error.MissingTensor, bind(alloc, &fewer));
}

test "the grid follows the reference's aspect-preserving search" {
    // The synthetic fixture: 96×64 → 84×56 px, 3×2 tokens.
    try std.testing.expectEqual(Grid{ .width_patches = 6, .height_patches = 4 }, gridFor(.{ .width = 96, .height = 64 }, max_tokens));
    // The aerial photo: 85×48 = 4,080 tokens under 4,096, and 21×12 = 252 at 256.
    try std.testing.expectEqual(Grid{ .width_patches = 170, .height_patches = 96 }, gridFor(.{ .width = 3840, .height = 2160 }, max_tokens));
    try std.testing.expectEqual(Grid{ .width_patches = 42, .height_patches = 24 }, gridFor(.{ .width = 3840, .height = 2160 }, 256));
    try std.testing.expectEqual(@as(usize, 1008), gridFor(.{ .width = 3840, .height = 2160 }, 1024).tokens());
    // A tall image, a tiny one (at least one token), and one too thin to fit.
    const tall = gridFor(.{ .width = 1000, .height = 3000 }, max_tokens);
    try std.testing.expect(tall.tokens() <= max_tokens and tall.height_patches > 2 * tall.width_patches);
    try std.testing.expectEqual(Grid{ .width_patches = 2, .height_patches = 2 }, gridFor(.{ .width = 5, .height = 5 }, max_tokens));
    try std.testing.expect(gridFor(.{ .width = 1, .height = 60000 }, 16).tokens() <= 16);
}

test "the window layout groups 32×32 patches, partial at the edges" {
    const alloc = std.testing.allocator;
    const grid: Grid = .{ .width_patches = 66, .height_patches = 34 };
    var layout = try Layout.init(alloc, grid);
    defer layout.deinit(alloc);
    // Three windows across (32, 32, 2), two down (32, 2).
    try std.testing.expectEqual(@as(usize, 6), layout.windows.len);
    const counts = [_]u32{ 1024, 1024, 64, 64, 64, 4 };
    var begin: u32 = 0;
    for (layout.windows, counts) |w, c| {
        try std.testing.expectEqual(Window{ .begin = begin, .count = c }, w);
        begin += c;
    }
    // Row 32 is the first patch of the window's second row: (0, 1) = 66.
    try std.testing.expectEqual(@as(u32, 66), layout.order[32]);
    // The third window starts at patch (64, 0), then (65, 0), then (64, 1).
    try std.testing.expectEqualSlices(u32, &.{ 64, 65, 130 }, layout.order[2048..2051]);
    // The last window: patches (64, 32), (65, 32), (64, 33), (65, 33).
    try std.testing.expectEqualSlices(u32, &.{ 32 * 66 + 64, 32 * 66 + 65, 33 * 66 + 64, 33 * 66 + 65 }, layout.order[layout.order.len - 4 ..]);
    for (layout.order, 0..) |p, r| try std.testing.expectEqual(@as(u32, @intCast(r)), layout.row_of[p]);
}

test "the rope table turns the first half by the column and the second by the row" {
    const alloc = std.testing.allocator;
    const grid: Grid = .{ .width_patches = 4, .height_patches = 2 };
    var layout = try Layout.init(alloc, grid);
    defer layout.deinit(alloc);
    var table: [8 * rope_pairs * 2]f32 = undefined;
    try ropeTable(&table, grid, layout);
    // Row 6 is patch (2, 1): column 3, row 2 (1-indexed); pair 0 and 24 at base^0.
    const row = table[6 * rope_pairs * 2 ..];
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(@cos(3.0))), row[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(@sin(2.0))), row[24 * 2 + 1], 1e-7);
    // Pair 1 turns at 10000^(−1/24).
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(@sin(3.0 * std.math.pow(f64, 10000, -1.0 / 24.0)))), row[3], 1e-7);
}

test "the pixel shuffle interleaves the four patches channel-major" {
    const alloc = std.testing.allocator;
    const grid: Grid = .{ .width_patches = 4, .height_patches = 2 };
    var layout = try Layout.init(alloc, grid);
    defer layout.deinit(alloc);
    const rows = try alloc.alloc(f32, grid.patches() * hidden);
    defer alloc.free(rows);
    // Row r carries its patch index in channel 0 and the channel in the rest.
    for (0..grid.patches()) |r| for (0..hidden) |c| {
        rows[r * hidden + c] = if (c == 0) @floatFromInt(layout.order[r]) else @floatFromInt(c);
    };
    const out = try alloc.alloc(f32, grid.tokens() * merged_width);
    defer alloc.free(out);
    try shuffle(rows, grid, layout, out);
    // Token 1 covers patches (2, 0), (3, 0), (2, 1), (3, 1) = 2, 3, 6, 7.
    try std.testing.expectEqualSlices(f32, &.{ 2, 3, 6, 7 }, out[merged_width..][0..4]);
    try std.testing.expectEqualSlices(f32, &.{ 5, 5, 5, 5 }, out[merged_width + 5 * 4 ..][0..4]);
}

test "the Lanczos stretch and F16 patches of the fixture equal the reference's im2col" {
    const alloc = std.testing.allocator;
    var source = try @import("image.zig").decodePpm(alloc, @embedFile("fixtures/synthetic-96x64.ppm"));
    defer source.deinit(alloc);
    const grid = gridFor(.{ .width = source.width, .height = source.height }, max_tokens);
    const target: preprocess.Size = .{ .width = grid.width_patches * patch, .height = grid.height_patches * patch };
    const resized = try preprocess.resizeLanczos(alloc, source, target);
    defer alloc.free(resized);
    var p = try preprocess.patches(alloc, resized, target, patchOptions(.{ 0.5, 0.5, 0.5 }, .{ 0.5, 0.5, 0.5 }));
    defer p.deinit(alloc);
    const expected = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(@embedFile("fixtures/muse-glimmer-synthetic/patches.f32"))));
    try std.testing.expectEqual(expected.len, p.values.len);
    var mismatches: usize = 0;
    for (expected, p.values) |want, got| {
        if (want != got) mismatches += 1;
    }
    try std.testing.expectEqual(@as(usize, 0), mismatches);
}
