//! The Inkling 2 projector (`clip.projector_type = inkling2v`): a 23-block
//! pre-LayerNorm ViT over 20-pixel patches with a class token, learned
//! positions (a 32×32 table resized to the grid) and a 2D rotary turn of the
//! queries and keys, layer scale on both residual branches, then the patch
//! rows merged 2×2 into a two-layer MLP at the language model's width. The
//! image is padded, never resized, to a multiple of 40 pixels unless it would
//! exceed the token cap. Row 0 is the class token; patch rows follow in
//! raster order. `bind` validates the companion file; `Runtime` is the CPU
//! reference (F64 accumulation), whose host tables the Metal plan shares.
//! Facts and provenance: docs/engine/vision.md § Inkling 2's projector.
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const weights = @import("../runtime/weights.zig");
const cpu = @import("../backends/cpu/root.zig");
const preprocess = @import("preprocess.zig");
const image = @import("image.zig");
const inventory = @import("../models/inventory.zig");
const qwen3vl = @import("qwen3vl.zig");
const rowsTimes = @import("rows_cpu.zig").rowsTimes;

const Tensor = gguf.Tensor;
/// The Metal plan, the same schedule on the device.
pub const Plan = @import("inkling2v_metal.zig").Plan;

pub const projector_type = "inkling2v";
pub const hidden = 1024;
pub const ffn = 4096;
pub const heads = 16;
pub const head_dim = 64;
pub const blocks = 23;
pub const patch = 20;
pub const merge = 2;
pub const output_width = 2304;
pub const merged_width = hidden * merge * merge;
pub const patch_values = patch * patch * 3;
/// Pixels per token side: one merge block of patches.
pub const token_side = patch * merge;
/// The learned position table is a `position_side²` grid.
pub const position_side = 32;
pub const norm_epsilon: f32 = 1e-5;
pub const rope_base: f64 = 10000;
pub const rope_pairs = head_dim / 2;
/// The format's token bounds per image.
pub const min_tokens = 1;
pub const max_tokens = 4096;
pub const max_patches = max_tokens * merge * merge;
/// The class row and every patch row.
pub const max_rows = max_patches + 1;
/// A padding pixel, -1 before the normalization.
pub const pad_value: f32 = -1;

pub const Error = qwen3vl.Error;
pub const Norm = qwen3vl.Norm;
pub const Linear = qwen3vl.Linear;
pub const Block = struct {
    norm1: Norm,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    /// The attention branch's per-channel scale.
    scale1: *const Tensor,
    norm2: Norm,
    up: Linear,
    down: Linear,
    /// The feed-forward branch's per-channel scale.
    scale2: *const Tensor,
};

pub const Binding = struct {
    class_embedding: *const Tensor,
    class_position: *const Tensor,
    /// `[20, 20, 3, hidden]` F16, columns `c · 400 + ky · 20 + kx`; no bias.
    patch_embedding: *const Tensor,
    /// `[hidden, position_side²]` F32.
    position_embedding: *const Tensor,
    pre_norm: Norm,
    layers: [blocks]Block,
    /// `4096 → 4096 → 2304`, no biases.
    adapter: [2]*const Tensor,
    mean: [3]f32,
    std: [3]f32,
    tensors: u32,
    bytes: u64,
};

/// Whether `doc` is an Inkling 2 projector; this file names the type under
/// `clip.vision.`, others at `clip.`.
pub fn matches(doc: *const gguf.Document) bool {
    for ([_][]const u8{ "clip.vision.projector_type", "clip.projector_type" }) |key| {
        const kind = qwen3vl.stringValue(doc, key) catch continue;
        return std.mem.eql(u8, kind, projector_type);
    }
    return false;
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
        .{ .key = "clip.vision.spatial_merge_size", .value = merge },
    };
    for (settings) |s| if (try qwen3vl.unsignedValue(doc, s.key) != s.value) return error.UnsupportedConfiguration;
    switch (doc.get("clip.vision.attention.layer_norm_epsilon") orelse return error.MissingMetadata) {
        .float => |f| if (@abs(f - norm_epsilon) > 1e-9) return error.UnsupportedConfiguration,
        else => return error.InvalidMetadata,
    }
}

/// Encodings each tensor class may have: the file's own; anything else is
/// refused.
const matrix_encodings = [_]u32{1}; // F16
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
        return .{ .weight = try self.named(prefix ++ ".weight", args, &.{ columns, rows }, &matrix_encodings), .bias = try self.named(prefix ++ ".bias", args, &.{rows}, &vector_only) };
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
    result.class_embedding = try binder.take("v.class_embd", &.{hidden}, &vector_only);
    result.class_position = try binder.take("v.class_position_embd.weight", &.{hidden}, &vector_only);
    result.patch_embedding = try binder.take("v.patch_embd.weight", &.{ patch, patch, 3, hidden }, &matrix_encodings);
    result.position_embedding = try binder.take("v.position_embd.weight", &.{ hidden, position_side * position_side }, &vector_only);
    result.pre_norm = try binder.norm("v.pre_ln", .{});
    for (&result.layers, 0..) |*layer, i| layer.* = .{
        .norm1 = try binder.norm("v.blk.{d}.ln1", .{i}),
        .query = try binder.linear("v.blk.{d}.attn_q", .{i}, hidden, hidden),
        .key = try binder.linear("v.blk.{d}.attn_k", .{i}, hidden, hidden),
        .value = try binder.linear("v.blk.{d}.attn_v", .{i}, hidden, hidden),
        .output = try binder.linear("v.blk.{d}.attn_out", .{i}, hidden, hidden),
        .scale1 = try binder.named("v.blk.{d}.ls1.weight", .{i}, &.{hidden}, &vector_only),
        .norm2 = try binder.norm("v.blk.{d}.ln2", .{i}),
        .up = try binder.linear("v.blk.{d}.ffn_up", .{i}, hidden, ffn),
        .down = try binder.linear("v.blk.{d}.ffn_down", .{i}, ffn, hidden),
        .scale2 = try binder.named("v.blk.{d}.ls2.weight", .{i}, &.{hidden}, &vector_only),
    };
    result.adapter = .{
        try binder.take("mm.1.weight", &.{ merged_width, merged_width }, &matrix_encodings),
        try binder.take("mm.2.weight", &.{ merged_width, output_width }, &matrix_encodings),
    };
    if (binder.remaining.count() != 0) return error.UnexpectedTensor;
    result.tensors = binder.tensors;
    result.bytes = binder.bytes;
    return result;
}

/// The patch grid of a prepared image; both sides are even.
pub const Grid = struct {
    width_patches: u32,
    height_patches: u32,
    pub fn patches(self: Grid) usize {
        return @as(usize, self.width_patches) * self.height_patches;
    }
    pub fn rows(self: Grid) usize {
        return self.patches() + 1;
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

/// Where an image's pixels go: `content` is the image's own size, or the
/// size it is resized to when its padded grid would pass `max` tokens; the
/// grid pads `content` up to whole tokens.
pub const Layout = struct { content: preprocess.Size, grid: Grid };

/// The format's rule: pad each side up to a multiple of 40. Only when
/// that passes `max` tokens is the image resized, keeping its aspect ratio:
/// each side rounded to the nearest 40 (at least 40), and if that still
/// passes the cap, `max(40, ⌊side / β / 40⌋ · 40)` with β = √(w·h / max
/// pixels), in F32; then nothing is padded.
pub fn layoutFor(size: preprocess.Size, max: u32) Layout {
    const side: u32 = token_side;
    const w = @max(1, size.width);
    const h = @max(1, size.height);
    const padded_w = std.math.divCeil(u32, w, side) catch unreachable;
    const padded_h = std.math.divCeil(u32, h, side) catch unreachable;
    if (@as(u64, padded_w) * padded_h <= max) return .{ .content = .{ .width = w, .height = h }, .grid = .{ .width_patches = padded_w * merge, .height_patches = padded_h * merge } };
    const side_f: f32 = @floatFromInt(side);
    const wf: f32 = @floatFromInt(w);
    const hf: f32 = @floatFromInt(h);
    var tw: u32 = @max(1, @as(u32, @intFromFloat(@round(wf / side_f))));
    var th: u32 = @max(1, @as(u32, @intFromFloat(@round(hf / side_f))));
    if (@as(u64, tw) * th > max) {
        const max_pixels: f32 = @floatFromInt(@as(u64, max) * side * side);
        const beta = @sqrt(hf * wf / max_pixels);
        tw = @max(1, @as(u32, @intFromFloat(@floor(wf / beta / side_f))));
        th = @max(1, @as(u32, @intFromFloat(@floor(hf / beta / side_f))));
    }
    return .{ .content = .{ .width = tw * side, .height = th * side }, .grid = .{ .width_patches = tw * merge, .height_patches = th * merge } };
}

/// The patch rows the kernel reads, raster order: each `c · 400 + ky · 20 +
/// kx`, the pixel normalized as `(v − 255·mean) / (255·std)` and rounded
/// to F16, the patch input's precision; a pixel past
/// `content` is `pad_value` before the normalization. `pixels` is `content`
/// RGB8. Caller owns the result.
pub fn patchRows(alloc: std.mem.Allocator, pixels: []const u8, content: preprocess.Size, grid: Grid, mean: [3]f32, std_dev: [3]f32) !preprocess.Patches {
    const width = grid.width_patches * patch;
    const height = grid.height_patches * patch;
    if (content.width > width or content.height > height or pixels.len != @as(usize, content.width) * content.height * 3) return error.InvalidShape;
    const n = grid.patches();
    if (n == 0 or n > max_patches) return error.InvalidShape;
    const values = try alloc.alloc(f32, n * patch_values);
    errdefer alloc.free(values);
    var m: [3]f32 = undefined;
    var s: [3]f32 = undefined;
    for (0..3) |c| {
        m[c] = mean[c] * 255.0;
        s[c] = std_dev[c] * 255.0;
    }
    for (0..n) |t| {
        const px = t % grid.width_patches;
        const py = t / grid.width_patches;
        const out = values[t * patch_values ..][0..patch_values];
        for (0..3) |c| for (0..patch) |ky| for (0..patch) |kx| {
            const x = px * patch + kx;
            const y = py * patch + ky;
            const raw: f32 = if (x < content.width and y < content.height) @floatFromInt(pixels[(y * content.width + x) * 3 + c]) else pad_value;
            out[c * patch * patch + ky * patch + kx] = @floatCast(@as(f16, @floatCast((raw - m[c]) / s[c])));
        };
    }
    return .{ .width_patches = grid.width_patches, .height_patches = grid.height_patches, .row = patch_values, .values = values };
}

/// Lays `source` out for `layout`: resized (Pillow's bicubic) only when the
/// layout says so, then padded into patch rows. Caller owns the result.
pub fn prepare(alloc: std.mem.Allocator, source: image.Rgb8, layout: Layout, mean: [3]f32, std_dev: [3]f32) !preprocess.Patches {
    if (layout.content.width == source.width and layout.content.height == source.height) return patchRows(alloc, source.pixels, layout.content, layout.grid, mean, std_dev);
    const resized = try preprocess.resizeBicubic(alloc, source, layout.content);
    defer alloc.free(resized);
    return patchRows(alloc, resized, layout.content, layout.grid, mean, std_dev);
}

/// The rotary table: per row `rope_pairs` (cos, sin) pairs for the
/// adjacent-pair rotation. Pair j < 16 turns channels (2j, 2j + 1) by the
/// column at `base^(−j/16)`, pair j ≥ 16 by the row at `base^(−(j−16)/16)`:
/// two half-width rotations. The class row sits at
/// (0, 0), patch (x, y) at (x + 1, y + 1). F64 angles rounded to F32.
pub fn ropeTable(out: []f32, grid: Grid) !void {
    const n = grid.rows();
    if (out.len < n * rope_pairs * 2) return error.InvalidShape;
    const half = rope_pairs / 2;
    for (0..n) |r| {
        const column: f64 = if (r == 0) 0 else @floatFromInt((r - 1) % grid.width_patches + 1);
        const row: f64 = if (r == 0) 0 else @floatFromInt((r - 1) / grid.width_patches + 1);
        for (0..rope_pairs) |j| {
            const position = if (j < half) column else row;
            const exponent = -@as(f64, @floatFromInt(j % half)) / @as(f64, @floatFromInt(half));
            const theta = position * std.math.pow(f64, rope_base, exponent);
            out[(r * rope_pairs + j) * 2] = @floatCast(@cos(theta));
            out[(r * rope_pairs + j) * 2 + 1] = @floatCast(@sin(theta));
        }
    }
}

/// What the encoder adds to each row before the pre-norm: the class
/// embedding plus its position on row 0; on patch rows the learned table
/// resized to the grid bilinearly with half-pixel centres and clamped edges
/// (no aligned corners), in F32, or the table itself when the grid is
/// 32×32. `out` holds `grid.rows() · hidden` values; `scratch` four rows of
/// `hidden`.
pub fn positionRows(view: weights.View, binding: *const Binding, grid: Grid, out: []f32, scratch: []f32) !void {
    const n = grid.patches();
    if (out.len < (n + 1) * hidden or scratch.len < 4 * hidden) return error.InvalidShape;
    try vectorRow(view, binding.class_embedding, out[0..hidden]);
    try vectorRow(view, binding.class_position, scratch[0..hidden]);
    for (out[0..hidden], scratch[0..hidden]) |*o, p| o.* += p;
    const table = binding.position_embedding;
    const side = position_side;
    for (0..n) |p| {
        const x: u32 = @intCast(p % grid.width_patches);
        const y: u32 = @intCast(p / grid.width_patches);
        const row = out[(p + 1) * hidden ..][0..hidden];
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
/// A 1-D F32 tensor of `out.len` values.
fn vectorRow(view: weights.View, tensor: *const Tensor, out: []f32) !void {
    const raw = try view.bytes(tensor);
    if (tensor.encoding_id != 0 or raw.len != out.len * 4) return error.InvalidShape;
    for (out, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
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
/// `c · 400 + ky · 20 + kx` columns zero-padded from 1,200 to `columns`.
/// Owned by the caller.
pub fn patchKernel(alloc: std.mem.Allocator, view: weights.View, binding: *const Binding, columns: usize) ![]f32 {
    if (columns < patch_values) return error.InvalidShape;
    const raw = try view.bytes(binding.patch_embedding);
    if (raw.len != hidden * patch_values * 2) return error.InvalidShape;
    const out = try alloc.alloc(f32, hidden * columns);
    @memset(out, 0);
    // GGUF dimensions [kx, ky, c, o]: kx fastest, so row o is contiguous.
    for (0..hidden) |o| for (0..patch_values) |i| {
        const bits = std.mem.readInt(u16, raw[(o * patch_values + i) * 2 ..][0..2], .little);
        out[o * columns + i] = @floatCast(@as(f16, @bitCast(bits)));
    };
    return out;
}

/// The 2×2 merge: merged token o of the `grid.widthTokens()`-wide token
/// grid holds, at element `s · hidden + c`, channel c of its patch s =
/// `2·dy + dx` (patch `(2·ox + dx, 2·oy + dy)`), read from the raster patch
/// rows (the class row excluded). `out` holds `grid.tokens() ·
/// merged_width` values.
pub fn merge2x2(patch_rows: []const f32, grid: Grid, out: []f32) !void {
    const tokens = grid.tokens();
    if (patch_rows.len < grid.patches() * hidden or out.len < tokens * merged_width) return error.InvalidShape;
    const wt = grid.widthTokens();
    for (0..tokens) |o| {
        const oy = o / wt;
        const ox = o % wt;
        for (0..merge * merge) |s| {
            const p = (oy * merge + s / merge) * grid.width_patches + ox * merge + s % merge;
            @memcpy(out[o * merged_width + s * hidden ..][0..hidden], patch_rows[p * hidden ..][0..hidden]);
        }
    }
}

/// What an encode leaves: the feature rows, or a check's residual rows.
pub const Result = union(enum) { features: []f32, residual: []f32 };

/// The CPU reference: matrix products with F64 accumulation, LayerNorm and
/// attention in F64, the exact (erf) GELU.
pub const Runtime = struct {
    alloc: std.mem.Allocator,
    /// Runs the products' row tasks.
    io: std.Io,
    view: weights.View,
    binding: *const Binding,
    kernel: []f32,
    vectors: Vectors,

    const Pair = struct { weight: []f32, bias: []f32 };
    const BlockVectors = struct { n1: Pair, qb: []f32, kb: []f32, vb: []f32, ob: []f32, s1: []f32, n2: Pair, ub: []f32, db: []f32, s2: []f32 };
    const Vectors = struct { pre: Pair, layers: [blocks]BlockVectors };

    pub fn init(alloc: std.mem.Allocator, io: std.Io, view: weights.View, binding: *const Binding) !Runtime {
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
            .s1 = try l.one(layer.scale1),
            .n2 = try l.pair(layer.norm2),
            .ub = try l.one(layer.up.bias),
            .db = try l.one(layer.down.bias),
            .s2 = try l.one(layer.scale2),
        };
        const kernel = try patchKernel(alloc, view, binding, patch_values);
        loaded.clearRetainingCapacity();
        return .{ .alloc = alloc, .io = io, .view = view, .binding = binding, .kernel = kernel, .vectors = vectors };
    }
    pub fn deinit(self: *Runtime) void {
        const a = self.alloc;
        a.free(self.kernel);
        const v = self.vectors;
        for ([_][]f32{ v.pre.weight, v.pre.bias }) |x| a.free(x);
        for (v.layers) |b| for ([_][]f32{ b.n1.weight, b.n1.bias, b.qb, b.kb, b.vb, b.ob, b.s1, b.n2.weight, b.n2.bias, b.ub, b.db, b.s2 }) |x| a.free(x);
        self.* = undefined;
    }

    /// Encodes `patches` (raster rows from `patchRows`) of an image of
    /// `grid` into `out`: `grid.tokens() × output_width` feature rows.
    pub fn encode(self: *Runtime, patches: preprocess.Patches, grid: Grid, out: []f32) !void {
        if (out.len < grid.tokens() * output_width) return error.InvalidShape;
        return self.run(patches, grid, blocks, .{ .features = out });
    }
    /// The residual rows (`grid.rows() × hidden`, the class row first)
    /// after the first `count` blocks, for the checks' traces.
    pub fn encodeResidual(self: *Runtime, patches: preprocess.Patches, grid: Grid, count: usize, rows: []f32) !void {
        if (count > blocks or rows.len < grid.rows() * hidden) return error.InvalidShape;
        return self.run(patches, grid, count, .{ .residual = rows });
    }

    fn run(self: *Runtime, patches: preprocess.Patches, grid: Grid, block_count: usize, result: Result) !void {
        const p = grid.patches();
        if (p == 0 or p > max_patches or patches.row != patch_values or patches.count() != p or patches.width_patches != grid.width_patches) return error.InvalidShape;
        const n = p + 1;
        const tokens = grid.tokens();
        const a = self.alloc;
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
        const up = try a.alloc(f32, @max(n * ffn, tokens * 2 * merged_width));
        defer a.free(up);
        const rope = try a.alloc(f32, n * rope_pairs * 2);
        defer a.free(rope);
        const scratch = try a.alloc(f32, cpu.matvecScratch(@max(ffn, merged_width)) + 4 * hidden);
        defer a.free(scratch);
        const scores = try a.alloc(f64, n);
        defer a.free(scores);
        try ropeTable(rope, grid);

        // Row 0 the class token, then the embedded patches; plus positions.
        @memset(x[0..hidden], 0);
        const kernel: cpu.Matrix = .{ .encoding = 0, .rows = hidden, .columns = patch_values, .bytes = std.mem.sliceAsBytes(self.kernel) };
        try rowsTimes(self.io, kernel, patches.values, p, x[hidden..], null, scratch);
        try positionRows(self.view, self.binding, grid, h, scratch);
        for (x, h) |*e, pos| e.* += pos;
        const vec = self.vectors;
        for (0..n) |t| qwen3vl.layerNormEpsilon(x[t * hidden ..][0..hidden], x[t * hidden ..][0..hidden], vec.pre.weight, vec.pre.bias, norm_epsilon);

        for (self.binding.layers[0..block_count], vec.layers[0..block_count]) |layer, lv| {
            for (0..n) |t| qwen3vl.layerNormEpsilon(x[t * hidden ..][0..hidden], h[t * hidden ..][0..hidden], lv.n1.weight, lv.n1.bias, norm_epsilon);
            try rowsTimes(self.io, try self.view.matrix(layer.query.weight), h, n, q, lv.qb, scratch);
            try rowsTimes(self.io, try self.view.matrix(layer.key.weight), h, n, k, lv.kb, scratch);
            try rowsTimes(self.io, try self.view.matrix(layer.value.weight), h, n, v, lv.vb, scratch);
            for (0..n) |r| {
                const table = rope[r * rope_pairs * 2 ..][0 .. rope_pairs * 2];
                for (0..heads) |head| {
                    rotate(q[r * hidden + head * head_dim ..][0..head_dim], table);
                    rotate(k[r * hidden + head * head_dim ..][0..head_dim], table);
                }
            }
            attention(q, k, v, attn, n, scores);
            try rowsTimes(self.io, try self.view.matrix(layer.output.weight), attn, n, h, lv.ob, scratch);
            for (0..n) |r| for (x[r * hidden ..][0..hidden], h[r * hidden ..][0..hidden], lv.s1) |*e, o, s| {
                e.* += o * s;
            };
            for (0..n) |t| qwen3vl.layerNormEpsilon(x[t * hidden ..][0..hidden], h[t * hidden ..][0..hidden], lv.n2.weight, lv.n2.bias, norm_epsilon);
            const u = up[0 .. n * ffn];
            try rowsTimes(self.io, try self.view.matrix(layer.up.weight), h, n, u, lv.ub, scratch);
            for (u) |*e| e.* = cpu.geluErf(e.*);
            try rowsTimes(self.io, try self.view.matrix(layer.down.weight), u, n, h, lv.db, scratch);
            for (0..n) |r| for (x[r * hidden ..][0..hidden], h[r * hidden ..][0..hidden], lv.s2) |*e, o, s| {
                e.* += o * s;
            };
        }
        const out = switch (result) {
            .residual => |rows| {
                @memcpy(rows[0 .. n * hidden], x);
                return;
            },
            .features => |f| f,
        };

        const merged = up[0 .. tokens * merged_width];
        try merge2x2(x[hidden..], grid, merged);
        const mid = up[tokens * merged_width ..][0 .. tokens * merged_width];
        try rowsTimes(self.io, try self.view.matrix(self.binding.adapter[0]), merged, tokens, mid, null, scratch);
        for (mid) |*e| e.* = cpu.geluErf(e.*);
        try rowsTimes(self.io, try self.view.matrix(self.binding.adapter[1]), mid, tokens, out[0 .. tokens * output_width], null, scratch);
    }
};

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

/// Bidirectional attention among all `n` rows, scale 1/√64.
fn attention(q: []const f32, k: []const f32, v: []const f32, out: []f32, n: usize, scores: []f64) void {
    const scale = 1.0 / @sqrt(@as(f64, head_dim));
    for (0..heads) |head| {
        const off = head * head_dim;
        for (0..n) |i| {
            const query = q[i * hidden + off ..][0..head_dim];
            var max: f64 = -std.math.inf(f64);
            for (0..n) |j| {
                const key = k[j * hidden + off ..][0..head_dim];
                var dot: f64 = 0;
                for (query, key) |a, b| dot += @as(f64, a) * b;
                scores[j] = dot * scale;
                max = @max(max, scores[j]);
            }
            var denominator: f64 = 0;
            for (scores[0..n]) |*s| {
                s.* = @exp(s.* - max);
                denominator += s.*;
            }
            const o = out[i * hidden + off ..][0..head_dim];
            for (o, 0..) |*e, d| {
                var sum: f64 = 0;
                for (0..n) |j| sum += scores[j] * v[j * hidden + off + d];
                e.* = @floatCast(sum / denominator);
            }
        }
    }
}

pub fn inventoryDocument(gpa: std.mem.Allocator) !gguf.Document {
    return inventory.document(gpa, @embedFile("fixtures/inkling2-mmproj.json"));
}

test "bind accepts the pinned projector inventory and refuses another encoding" {
    const alloc = std.testing.allocator;
    var doc = try inventoryDocument(alloc);
    defer doc.storage.deinit();
    const binding = try bind(alloc, &doc);
    try std.testing.expectEqual(@as(u32, 422), binding.tensors);
    try std.testing.expectEqual([3]f32{ 0.5, 0.5, 0.5 }, binding.mean);
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

test "the layout pads to whole tokens and resizes only past the cap" {
    // The synthetic fixture: 96×64 pads to 120×80, 3×2 tokens.
    try std.testing.expectEqual(Layout{ .content = .{ .width = 96, .height = 64 }, .grid = .{ .width_patches = 6, .height_patches = 4 } }, layoutFor(.{ .width = 96, .height = 64 }, max_tokens));
    // Exact multiples gain no padding; a 1-pixel image is one token.
    try std.testing.expectEqual(@as(usize, 256), layoutFor(.{ .width = 640, .height = 640 }, max_tokens).grid.tokens());
    try std.testing.expectEqual(@as(usize, 1), layoutFor(.{ .width = 1, .height = 1 }, max_tokens).grid.tokens());
    // 1200×630 pads to 30×16 = 480 tokens.
    try std.testing.expectEqual(Grid{ .width_patches = 60, .height_patches = 32 }, layoutFor(.{ .width = 1200, .height = 630 }, max_tokens).grid);
    // Past the cap the image is resized to whole tokens under it.
    const big = layoutFor(.{ .width = 3840, .height = 2160 }, max_tokens);
    try std.testing.expect(big.grid.tokens() <= max_tokens and big.content.width == big.grid.width_patches * patch and big.content.height == big.grid.height_patches * patch);
    try std.testing.expect(layoutFor(.{ .width = 3840, .height = 2160 }, 64).grid.tokens() <= 64);
}

test "the patch rows pad with the normalized fill and round to F16" {
    const alloc = std.testing.allocator;
    // A 2×1 image of one grey and one white pixel, padded to one token.
    const pixels = [_]u8{ 128, 128, 128, 255, 255, 255 };
    const layout = layoutFor(.{ .width = 2, .height = 1 }, max_tokens);
    var rows = try patchRows(alloc, &pixels, layout.content, layout.grid, .{ 0.5, 0.5, 0.5 }, .{ 0.5, 0.5, 0.5 });
    defer rows.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 4), rows.count());
    const half = struct {
        fn of(v: f32) f32 {
            return @floatCast(@as(f16, @floatCast(v)));
        }
    };
    try std.testing.expectEqual(half.of((128 - 127.5) / 127.5), rows.values[0]);
    try std.testing.expectEqual(half.of(1), rows.values[1]);
    try std.testing.expectEqual(half.of((-1 - 127.5) / 127.5), rows.values[2]);
    // Patch 3 is all padding.
    try std.testing.expectEqual(half.of((-1 - 127.5) / 127.5), rows.values[3 * patch_values + 2 * 400 + 399]);
}

test "the rope table leaves the class row unturned and counts patches from 1" {
    const grid: Grid = .{ .width_patches = 4, .height_patches = 2 };
    var table: [9 * rope_pairs * 2]f32 = undefined;
    try ropeTable(&table, grid);
    for (0..rope_pairs) |j| try std.testing.expectEqual(@as(f32, 1), table[j * 2]);
    // Row 7 is patch 6 = (2, 1): column 3, row 2; pair 0 and 16 at base^0.
    const row = table[7 * rope_pairs * 2 ..];
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(@cos(3.0))), row[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(@sin(2.0))), row[16 * 2 + 1], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(@sin(3.0 * std.math.pow(f64, 10000, -1.0 / 16.0)))), row[3], 1e-7);
}

test "the merge lays the four patches position-major" {
    const alloc = std.testing.allocator;
    const grid: Grid = .{ .width_patches = 4, .height_patches = 2 };
    const rows = try alloc.alloc(f32, grid.patches() * hidden);
    defer alloc.free(rows);
    for (0..grid.patches()) |p| for (0..hidden) |c| {
        rows[p * hidden + c] = if (c == 0) @floatFromInt(p) else @floatFromInt(c);
    };
    const out = try alloc.alloc(f32, grid.tokens() * merged_width);
    defer alloc.free(out);
    try merge2x2(rows, grid, out);
    // Token 1 covers patches (2, 0), (3, 0), (2, 1), (3, 1) = 2, 3, 6, 7.
    for ([_]f32{ 2, 3, 6, 7 }, 0..) |want, s| try std.testing.expectEqual(want, out[merged_width + s * hidden]);
    try std.testing.expectEqual(@as(f32, 5), out[merged_width + 2 * hidden + 5]);
}
