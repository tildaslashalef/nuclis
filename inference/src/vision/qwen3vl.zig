//! The Qwen3-VL projector (`clip.projector_type = qwen3vl_merger`): a
//! SigLIP-style encoder of 27 pre-LayerNorm blocks with 2-D RoPE, then a
//! 2×2 spatial merge into a two-layer MLP at the language model's width.
//! `bind` validates the companion file; `Runtime` is the CPU reference
//! (F64 accumulation); the host-side tables (positions, RoPE, the summed
//! patch kernel) are shared with the Metal plan. Facts, provenance, and the
//! trace: docs/reference/vision.md § The Qwen3-VL projector.
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const weights = @import("../runtime/weights.zig");
const cpu = @import("../backends/cpu/root.zig");
const preprocess = @import("preprocess.zig");
const inventory = @import("../models/inventory.zig");

const Tensor = gguf.Tensor;
/// The Metal plan, the same schedule on the device.
pub const Plan = @import("qwen3vl_metal.zig").Plan;

pub const architecture = "clip";
pub const projector_type = "qwen3vl_merger";
pub const hidden = 1152;
pub const ffn = 4304;
pub const heads = 16;
pub const head_dim = 72;
pub const blocks = 27;
pub const patch = 16;
pub const merge = 2;
pub const output_width = 5120;
/// The learned position table is a `position_side²` grid, bilinearly
/// resized (corners aligned) to the image's patch grid.
pub const position_side = 48;
pub const patch_values = patch * patch * 3;
pub const merged_width = hidden * merge * merge;
pub const qkv_width = 3 * hidden;
pub const rope_base: f64 = 10000;
pub const norm_epsilon: f32 = 1e-6;
/// Pixels per output token: one merge block of patches.
pub const token_pixels = patch * patch * merge * merge;
/// Output tokens per image, the host bound: the reference's minimum, and a
/// maximum that keeps one image at 4,096 patches (the bidirectional
/// attention's row limit); larger images are scaled down to fit.
pub const min_tokens = 8;
pub const max_tokens = 1024;
pub const max_patches = max_tokens * merge * merge;

pub const Error = error{
    MissingMetadata,
    InvalidMetadata,
    UnsupportedArchitecture,
    UnsupportedProjector,
    UnsupportedConfiguration,
    MissingTensor,
    DuplicateTensor,
    UnexpectedTensor,
    InvalidTensorShape,
    UnsupportedTensorEncoding,
    OutOfMemory,
};

pub const Norm = struct { weight: *const Tensor, bias: *const Tensor };
pub const Linear = struct { weight: *const Tensor, bias: *const Tensor };
pub const Block = struct { norm1: Norm, qkv: Linear, output: Linear, norm2: Norm, up: Linear, down: Linear };

pub const Binding = struct {
    /// The two temporal slices of the 3-D patch convolution; a still image is
    /// both frames, so the slices are summed at load.
    patch_embedding: [2]*const Tensor,
    patch_bias: *const Tensor,
    /// `[position_side², hidden]`.
    position_embedding: *const Tensor,
    layers: [blocks]Block,
    post_norm: Norm,
    merger_0: Linear,
    merger_2: Linear,
    mean: [3]f32,
    std: [3]f32,
    tensors: u32,
    bytes: u64,
};

pub fn stringValue(doc: *const gguf.Document, key: []const u8) Error![]const u8 {
    return switch (doc.get(key) orelse return error.MissingMetadata) {
        .string => |s| s,
        else => error.InvalidMetadata,
    };
}
pub fn unsignedValue(doc: *const gguf.Document, key: []const u8) Error!u64 {
    return switch (doc.get(key) orelse return error.MissingMetadata) {
        .unsigned => |n| n,
        .signed => |n| if (n >= 0) @intCast(n) else error.InvalidMetadata,
        else => error.InvalidMetadata,
    };
}
pub fn floatTriple(doc: *const gguf.Document, key: []const u8) Error![3]f32 {
    const array = switch (doc.get(key) orelse return error.MissingMetadata) {
        .array => |a| a,
        else => return error.InvalidMetadata,
    };
    if (array.element_type != .float32 or array.count != 3) return error.UnsupportedConfiguration;
    const values = array.values orelse return error.InvalidMetadata;
    var out: [3]f32 = undefined;
    for (&out, values) |*o, v| o.* = switch (v) {
        .float => |f| @floatCast(f),
        else => return error.InvalidMetadata,
    };
    return out;
}

fn validateMetadata(doc: *const gguf.Document) Error!void {
    if (!std.mem.eql(u8, try stringValue(doc, "general.architecture"), architecture)) return error.UnsupportedArchitecture;
    if (!std.mem.eql(u8, try stringValue(doc, "clip.projector_type"), projector_type)) return error.UnsupportedProjector;
    const settings = [_]struct { key: []const u8, value: u64 }{
        .{ .key = "clip.vision.projection_dim", .value = output_width },
        .{ .key = "clip.vision.image_size", .value = 768 },
        .{ .key = "clip.vision.patch_size", .value = patch },
        .{ .key = "clip.vision.embedding_length", .value = hidden },
        .{ .key = "clip.vision.feed_forward_length", .value = ffn },
        .{ .key = "clip.vision.block_count", .value = blocks },
        .{ .key = "clip.vision.attention.head_count", .value = heads },
        .{ .key = "clip.vision.spatial_merge_size", .value = merge },
    };
    for (settings) |s| if (try unsignedValue(doc, s.key) != s.value) return error.UnsupportedConfiguration;
    switch (doc.get("clip.use_gelu") orelse return error.MissingMetadata) {
        .boolean => |b| if (!b) return error.UnsupportedConfiguration,
        else => return error.InvalidMetadata,
    }
    switch (doc.get("clip.vision.attention.layer_norm_epsilon") orelse return error.MissingMetadata) {
        .float => |f| if (@abs(f - 1e-6) > 1e-9) return error.UnsupportedConfiguration,
        else => return error.InvalidMetadata,
    }
    // Deepstack layers would add feature rows the language model sums into
    // its early layers; this file has none, and the graph does not handle them.
    if (doc.get("clip.vision.is_deepstack_layers")) |value| switch (value) {
        .array => |a| if (a.values) |values| for (values) |v| switch (v) {
            .boolean => |b| if (b) return error.UnsupportedConfiguration,
            .unsigned => |n| if (n != 0) return error.UnsupportedConfiguration,
            else => return error.InvalidMetadata,
        },
        else => return error.InvalidMetadata,
    };
}

pub const Binder = struct {
    remaining: std.StringHashMap(*const Tensor),
    tensors: u32 = 0,
    bytes: u64 = 0,

    pub fn take(self: *Binder, name: []const u8, dimensions: []const u64, f32_only: bool) Error!*const Tensor {
        const entry = self.remaining.fetchRemove(name) orelse return error.MissingTensor;
        const tensor = entry.value;
        if (!std.mem.eql(u64, tensor.dimensions, dimensions)) return error.InvalidTensorShape;
        // Weights are F32 or BF16 (the generic decoders on both executors);
        // vectors are F32.
        const ok = if (f32_only) tensor.encoding_id == 0 else (tensor.encoding_id == 0 or tensor.encoding_id == 30);
        if (!ok) return error.UnsupportedTensorEncoding;
        self.tensors += 1;
        self.bytes = std.math.add(u64, self.bytes, tensor.bytes) catch return error.InvalidTensorShape;
        return tensor;
    }
    pub fn named(self: *Binder, comptime fmt: []const u8, args: anytype, dimensions: []const u64, f32_only: bool) Error!*const Tensor {
        var name: [96]u8 = undefined;
        return self.take(std.fmt.bufPrint(&name, fmt, args) catch unreachable, dimensions, f32_only);
    }
    pub fn norm(self: *Binder, comptime prefix: []const u8, args: anytype, width: u64) Error!Norm {
        return .{ .weight = try self.named(prefix ++ ".weight", args, &.{width}, true), .bias = try self.named(prefix ++ ".bias", args, &.{width}, true) };
    }
    pub fn linear(self: *Binder, comptime prefix: []const u8, args: anytype, columns: u64, rows: u64) Error!Linear {
        return .{ .weight = try self.named(prefix ++ ".weight", args, &.{ columns, rows }, false), .bias = try self.named(prefix ++ ".bias", args, &.{rows}, true) };
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
    result.mean = try floatTriple(doc, "clip.vision.image_mean");
    result.std = try floatTriple(doc, "clip.vision.image_std");
    for (result.std) |s| if (!(s > 0)) return error.UnsupportedConfiguration;
    result.patch_embedding = .{
        try binder.take("v.patch_embd.weight", &.{ patch, patch, 3, hidden }, true),
        try binder.take("v.patch_embd.weight.1", &.{ patch, patch, 3, hidden }, true),
    };
    result.patch_bias = try binder.take("v.patch_embd.bias", &.{hidden}, true);
    result.position_embedding = try binder.take("v.position_embd.weight", &.{ hidden, position_side * position_side }, true);
    for (&result.layers, 0..) |*layer, i| layer.* = .{
        .norm1 = try binder.norm("v.blk.{d}.ln1", .{i}, hidden),
        .qkv = try binder.linear("v.blk.{d}.attn_qkv", .{i}, hidden, qkv_width),
        .output = try binder.linear("v.blk.{d}.attn_out", .{i}, hidden, hidden),
        .norm2 = try binder.norm("v.blk.{d}.ln2", .{i}, hidden),
        .up = try binder.linear("v.blk.{d}.ffn_up", .{i}, hidden, ffn),
        .down = try binder.linear("v.blk.{d}.ffn_down", .{i}, ffn, hidden),
    };
    result.post_norm = try binder.norm("v.post_ln", .{}, hidden);
    result.merger_0 = try binder.linear("mm.0", .{}, merged_width, merged_width);
    result.merger_2 = try binder.linear("mm.2", .{}, merged_width, output_width);
    if (binder.remaining.count() != 0) return error.UnexpectedTensor;
    result.tensors = binder.tensors;
    result.bytes = binder.bytes;
    return result;
}

/// The grid of an image after the smart size: patches per side and output
/// tokens.
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

/// The patch grid of a decoded image of `size`, at most `max` tokens
/// (`min_tokens..max_tokens`).
pub fn gridFor(size: preprocess.Size, max: u32) Grid {
    const target = preprocess.smartSize(size, .{ .align_size = patch * merge, .min_pixels = min_tokens * token_pixels, .max_pixels = max * token_pixels });
    return .{ .width_patches = target.width / patch, .height_patches = target.height / patch };
}

/// Patch `t` of the merge walk: its coordinates on the patch grid.
pub fn patchPosition(t: usize, grid: Grid) struct { x: u32, y: u32 } {
    const block = t / (merge * merge);
    const inner = t % (merge * merge);
    const blocks_wide = grid.width_patches / merge;
    const x2: u32 = @intCast(block % blocks_wide);
    const y2: u32 = @intCast(block / blocks_wide);
    return .{ .x = x2 * merge + @as(u32, @intCast(inner % merge)), .y = y2 * merge + @as(u32, @intCast(inner / merge)) };
}

pub const rope_pairs = head_dim / 2;
/// The 2-D rotary table for a grid: per patch (merge order) `rope_pairs`
/// (cos, sin) pairs; pairs `[0, 18)` turn with the patch's y at frequencies
/// `base^(-2j/36)`, pairs `[18, 36)` with its x at the same frequencies
/// (the reference's `GGML_ROPE_TYPE_VISION`). `out` holds
/// `grid.patches() · rope_pairs · 2` values, F64 angles rounded to F32 as
/// the text tables are.
pub fn ropeTable(out: []f32, grid: Grid) !void {
    const n = grid.patches();
    if (out.len < n * rope_pairs * 2) return error.InvalidShape;
    const half = rope_pairs / 2;
    for (0..n) |t| {
        const p = patchPosition(t, grid);
        for (0..rope_pairs) |j| {
            const position: f64 = @floatFromInt(if (j < half) p.y else p.x);
            const exponent = -2.0 * @as(f64, @floatFromInt(j % half)) / @as(f64, @floatFromInt(rope_pairs));
            const theta = position * std.math.pow(f64, rope_base, exponent);
            out[(t * rope_pairs + j) * 2] = @floatCast(@cos(theta));
            out[(t * rope_pairs + j) * 2 + 1] = @floatCast(@sin(theta));
        }
    }
}

/// The learned position rows for a grid, in merge order: the 48×48 table
/// bilinearly resized to `width × height` patches with corners aligned
/// (the reference's `ggml_interpolate` in F32). `out` holds
/// `grid.patches() · hidden` values; `scratch` four rows of `hidden`.
pub fn positionRows(view: weights.View, table: *const Tensor, grid: Grid, out: []f32, scratch: []f32) !void {
    const n = grid.patches();
    if (out.len < n * hidden or scratch.len < 4 * hidden) return error.InvalidShape;
    const side = position_side;
    for (0..n) |t| {
        const p = patchPosition(t, grid);
        const row = out[t * hidden ..][0..hidden];
        if (grid.width_patches == side and grid.height_patches == side) {
            try view.row(table, @as(usize, p.y) * side + p.x, row);
            continue;
        }
        const sx = sample(p.x, grid.width_patches);
        const sy = sample(p.y, grid.height_patches);
        const corners = [4][2]usize{ .{ sx.first, sy.first }, .{ sx.second, sy.first }, .{ sx.first, sy.second }, .{ sx.second, sy.second } };
        for (corners, 0..) |c, k| try view.row(table, c[1] * side + c[0], scratch[k * hidden ..][0..hidden]);
        const a = scratch[0..hidden];
        const b = scratch[hidden..][0..hidden];
        const c = scratch[2 * hidden ..][0..hidden];
        const d = scratch[3 * hidden ..][0..hidden];
        for (row, 0..) |*o, i| o.* = a[i] * (1 - sx.frac) * (1 - sy.frac) + b[i] * sx.frac * (1 - sy.frac) + c[i] * (1 - sx.frac) * sy.frac + d[i] * sx.frac * sy.frac;
    }
}
/// One axis of the align-corners resize: source indices and the fraction.
fn sample(index: u32, count: u32) struct { first: usize, second: usize, frac: f32 } {
    const side: f32 = position_side;
    const scale: f32 = if (count > 1) (@as(f32, @floatFromInt(count)) - 1) / (side - 1) else @as(f32, @floatFromInt(count)) / side;
    const x = @as(f32, @floatFromInt(index)) / scale;
    const floor: i64 = @intFromFloat(@floor(x));
    const first: i64 = @max(0, @min(floor, position_side - 1));
    const second: i64 = @max(0, @min(floor + 1, position_side - 1));
    const frac = std.math.clamp(x - @as(f32, @floatFromInt(first)), 0, 1);
    return .{ .first = @intCast(first), .second = @intCast(second), .frac = frac };
}

/// The summed patch kernel `W0 + W1` as an F32 `[hidden × patch_values]`
/// matrix (row-major, `c · patch² + ky · patch + kx` columns, the layout
/// `preprocess.patches` writes). Owned by the caller.
pub fn patchKernel(alloc: std.mem.Allocator, view: weights.View, binding: *const Binding) ![]f32 {
    const out = try alloc.alloc(f32, hidden * patch_values);
    errdefer alloc.free(out);
    @memset(out, 0);
    for (binding.patch_embedding) |tensor| {
        const raw = try view.bytes(tensor);
        if (raw.len != hidden * patch_values * 4) return error.InvalidShape;
        // GGUF dimensions [kx, ky, c, o]: kx fastest, so element
        // (o, c, ky, kx) sits at ((o · 3 + c) · patch + ky) · patch + kx — the
        // same order as our columns.
        for (out, 0..) |*o, i| o.* += @as(f32, @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little)));
    }
    return out;
}

/// Sizes of the working buffers `encode` needs for `n` patches, shared with
/// the Metal plan's allocation.
pub fn scratchFloats(n: usize) usize {
    return n * (hidden * 3 + qkv_width + ffn) + n * n;
}

/// The CPU reference: every matrix product through `cpu.matvec` (F64
/// accumulation), LayerNorm and attention in F64, GELU the tanh form.
pub const Runtime = struct {
    alloc: std.mem.Allocator,
    view: weights.View,
    binding: *const Binding,
    kernel: []f32,
    vectors: Vectors,

    const BlockVectors = struct { n1w: []f32, n1b: []f32, qkvb: []f32, ob: []f32, n2w: []f32, n2b: []f32, ub: []f32, db: []f32 };
    const Vectors = struct { patch_bias: []f32, layers: [blocks]BlockVectors, post_w: []f32, post_b: []f32, m0b: []f32, m2b: []f32 };

    pub fn init(alloc: std.mem.Allocator, view: weights.View, binding: *const Binding) !Runtime {
        var arena_list: std.ArrayList([]f32) = .empty;
        defer arena_list.deinit(alloc);
        errdefer for (arena_list.items) |v| alloc.free(v);
        const V = struct {
            fn load(list: *std.ArrayList([]f32), a: std.mem.Allocator, v: weights.View, t: *const Tensor) ![]f32 {
                const out = try v.vector(a, t);
                errdefer a.free(out);
                try list.append(a, out);
                return out;
            }
        };
        var vectors: Vectors = undefined;
        vectors.patch_bias = try V.load(&arena_list, alloc, view, binding.patch_bias);
        for (&vectors.layers, binding.layers) |*out, layer| out.* = .{
            .n1w = try V.load(&arena_list, alloc, view, layer.norm1.weight),
            .n1b = try V.load(&arena_list, alloc, view, layer.norm1.bias),
            .qkvb = try V.load(&arena_list, alloc, view, layer.qkv.bias),
            .ob = try V.load(&arena_list, alloc, view, layer.output.bias),
            .n2w = try V.load(&arena_list, alloc, view, layer.norm2.weight),
            .n2b = try V.load(&arena_list, alloc, view, layer.norm2.bias),
            .ub = try V.load(&arena_list, alloc, view, layer.up.bias),
            .db = try V.load(&arena_list, alloc, view, layer.down.bias),
        };
        vectors.post_w = try V.load(&arena_list, alloc, view, binding.post_norm.weight);
        vectors.post_b = try V.load(&arena_list, alloc, view, binding.post_norm.bias);
        vectors.m0b = try V.load(&arena_list, alloc, view, binding.merger_0.bias);
        vectors.m2b = try V.load(&arena_list, alloc, view, binding.merger_2.bias);
        const kernel = try patchKernel(alloc, view, binding);
        errdefer alloc.free(kernel);
        arena_list.clearRetainingCapacity();
        return .{ .alloc = alloc, .view = view, .binding = binding, .kernel = kernel, .vectors = vectors };
    }
    pub fn deinit(self: *Runtime) void {
        const a = self.alloc;
        a.free(self.kernel);
        a.free(self.vectors.patch_bias);
        for (self.vectors.layers) |l| for ([_][]f32{ l.n1w, l.n1b, l.qkvb, l.ob, l.n2w, l.n2b, l.ub, l.db }) |v| a.free(v);
        for ([_][]f32{ self.vectors.post_w, self.vectors.post_b, self.vectors.m0b, self.vectors.m2b }) |v| a.free(v);
        self.* = undefined;
    }

    /// Encodes `patches` (a grid's rows from `preprocess.patches`) into
    /// `out`: `grid.tokens() × output_width` feature rows.
    pub fn encode(self: *Runtime, patches: preprocess.Patches, out: []f32) !void {
        const grid: Grid = .{ .width_patches = patches.width_patches, .height_patches = patches.height_patches };
        const n = grid.patches();
        if (n == 0 or n > max_patches or patches.row != patch_values or out.len < grid.tokens() * output_width) return error.InvalidShape;
        const a = self.alloc;
        const x = try a.alloc(f32, n * hidden);
        defer a.free(x);
        const h = try a.alloc(f32, n * hidden);
        defer a.free(h);
        const qkv = try a.alloc(f32, n * qkv_width);
        defer a.free(qkv);
        const attn = try a.alloc(f32, n * hidden);
        defer a.free(attn);
        const up = try a.alloc(f32, n * ffn);
        defer a.free(up);
        const rope = try a.alloc(f32, n * rope_pairs * 2);
        defer a.free(rope);
        const scratch = try a.alloc(f32, @max(merged_width, ffn) + 4 * hidden);
        defer a.free(scratch);
        const scores = try a.alloc(f64, n);
        defer a.free(scores);
        try ropeTable(rope, grid);

        // Patch embedding, then the interpolated learned positions.
        const kernel_matrix: cpu.Matrix = .{ .encoding = 0, .rows = hidden, .columns = patch_values, .bytes = std.mem.sliceAsBytes(self.kernel) };
        for (0..n) |t| {
            try cpu.matvec(kernel_matrix, patches.values[t * patch_values ..][0..patch_values], x[t * hidden ..][0..hidden], scratch);
            for (x[t * hidden ..][0..hidden], self.vectors.patch_bias) |*v, b| v.* += b;
        }
        try positionRows(self.view, self.binding.position_embedding, grid, h, scratch);
        for (x, h) |*v, p| v.* += p;

        for (self.binding.layers, self.vectors.layers) |layer, vec| {
            for (0..n) |t| layerNorm(x[t * hidden ..][0..hidden], h[t * hidden ..][0..hidden], vec.n1w, vec.n1b);
            const qkv_matrix = try self.view.matrix(layer.qkv.weight);
            for (0..n) |t| {
                const row = qkv[t * qkv_width ..][0..qkv_width];
                try cpu.matvec(qkv_matrix, h[t * hidden ..][0..hidden], row, scratch);
                for (row, vec.qkvb) |*v, b| v.* += b;
                const table = rope[t * rope_pairs * 2 ..][0 .. rope_pairs * 2];
                for (0..heads) |head| {
                    rotate(row[head * head_dim ..][0..head_dim], table);
                    rotate(row[hidden + head * head_dim ..][0..head_dim], table);
                }
            }
            try attention(qkv, attn, n, scores);
            const out_matrix = try self.view.matrix(layer.output.weight);
            for (0..n) |t| {
                const o = h[t * hidden ..][0..hidden];
                try cpu.matvec(out_matrix, attn[t * hidden ..][0..hidden], o, scratch);
                for (x[t * hidden ..][0..hidden], o, vec.ob) |*v, r, b| v.* += r + b;
            }
            const up_matrix = try self.view.matrix(layer.up.weight);
            const down_matrix = try self.view.matrix(layer.down.weight);
            for (0..n) |t| {
                layerNorm(x[t * hidden ..][0..hidden], h[t * hidden ..][0..hidden], vec.n2w, vec.n2b);
                const u = up[t * ffn ..][0..ffn];
                try cpu.matvec(up_matrix, h[t * hidden ..][0..hidden], u, scratch);
                for (u, vec.ub) |*v, b| v.* = cpu.gelu(v.* + b);
                const d = h[t * hidden ..][0..hidden];
                try cpu.matvec(down_matrix, u, d, scratch);
                for (x[t * hidden ..][0..hidden], d, vec.db) |*v, r, b| v.* += r + b;
            }
        }

        // Post-norm per patch, then each merge block's four rows as one.
        for (0..n) |t| layerNorm(x[t * hidden ..][0..hidden], h[t * hidden ..][0..hidden], self.vectors.post_w, self.vectors.post_b);
        const m0 = try self.view.matrix(self.binding.merger_0.weight);
        const m2 = try self.view.matrix(self.binding.merger_2.weight);
        const mid = up[0..merged_width];
        for (0..grid.tokens()) |g| {
            try cpu.matvec(m0, h[g * merged_width ..][0..merged_width], mid, scratch);
            for (mid, self.vectors.m0b) |*v, b| v.* = cpu.gelu(v.* + b);
            const row = out[g * output_width ..][0..output_width];
            try cpu.matvec(m2, mid, row, scratch);
            for (row, self.vectors.m2b) |*v, b| v.* += b;
        }
    }
};

/// LayerNorm with weight and bias, statistics in F64.
pub fn layerNorm(input: []const f32, output: []f32, weight: []const f32, bias: []const f32) void {
    layerNormEpsilon(input, output, weight, bias, norm_epsilon);
}
/// `layerNorm` with a caller's epsilon.
pub fn layerNormEpsilon(input: []const f32, output: []f32, weight: []const f32, bias: []const f32, epsilon: f32) void {
    var sum: f64 = 0;
    for (input) |v| sum += v;
    const mean = sum / @as(f64, @floatFromInt(input.len));
    var sq: f64 = 0;
    for (input) |v| {
        const d = v - mean;
        sq += d * d;
    }
    const scale = 1.0 / @sqrt(sq / @as(f64, @floatFromInt(input.len)) + @as(f64, epsilon));
    for (output, input, weight, bias) |*o, v, w, b| o.* = @floatCast((v - mean) * scale * w + b);
}

/// Rotates one head's `head_dim` values by the table's pairs `(j, j + 36)`.
fn rotate(head: []f32, table: []const f32) void {
    for (0..rope_pairs) |j| {
        const c: f64 = table[j * 2];
        const s: f64 = table[j * 2 + 1];
        const a: f64 = head[j];
        const b: f64 = head[j + rope_pairs];
        head[j] = @floatCast(a * c - b * s);
        head[j + rope_pairs] = @floatCast(a * s + b * c);
    }
}

/// Bidirectional attention over `n` rows of the fused `qkv` layout
/// (`[q heads | k heads | v heads]` per row); `out` gets `n × hidden`.
fn attention(qkv: []const f32, out: []f32, n: usize, scores: []f64) !void {
    const scale = 1.0 / @sqrt(@as(f64, head_dim));
    for (0..heads) |head| {
        for (0..n) |i| {
            const q = qkv[i * qkv_width + head * head_dim ..][0..head_dim];
            var max: f64 = -std.math.inf(f64);
            for (0..n) |j| {
                const k = qkv[j * qkv_width + hidden + head * head_dim ..][0..head_dim];
                var dot: f64 = 0;
                for (q, k) |a, b| dot += @as(f64, a) * b;
                scores[j] = dot * scale;
                max = @max(max, scores[j]);
            }
            var denominator: f64 = 0;
            for (scores[0..n]) |*s| {
                s.* = @exp(s.* - max);
                denominator += s.*;
            }
            const o = out[i * hidden + head * head_dim ..][0..head_dim];
            for (o, 0..) |*v, d| {
                var sum: f64 = 0;
                for (0..n) |j| sum += scores[j] * qkv[j * qkv_width + 2 * hidden + head * head_dim + d];
                v.* = @floatCast(sum / denominator);
            }
        }
    }
}

pub fn inventoryDocument(gpa: std.mem.Allocator) !gguf.Document {
    return inventory.document(gpa, @embedFile("fixtures/qwen35-mmproj.json"));
}

test "bind accepts the pinned projector inventory and reports its tensors" {
    var doc = try inventoryDocument(std.testing.allocator);
    defer doc.storage.deinit();
    const binding = try bind(std.testing.allocator, &doc);
    try std.testing.expectEqual(@as(u32, 334), binding.tensors);
    try std.testing.expectEqual([3]f32{ 0.5, 0.5, 0.5 }, binding.mean);
    try std.testing.expectEqual(@as(u64, 4608), binding.merger_0.weight.dimensions[0]);
}

test "bind rejects another projector type and a missing tensor" {
    const alloc = std.testing.allocator;
    var doc = try inventoryDocument(alloc);
    defer doc.storage.deinit();
    const meta = try alloc.dupe(gguf.Metadata, doc.metadata);
    defer alloc.free(meta);
    for (meta) |*m| if (std.mem.eql(u8, m.key, "clip.projector_type")) {
        m.value = .{ .string = "gemma4v" };
    };
    var other = doc;
    other.metadata = meta;
    try std.testing.expectError(error.UnsupportedProjector, bind(alloc, &other));
    var fewer = doc;
    fewer.tensors = doc.tensors[1..];
    try std.testing.expectError(error.MissingTensor, bind(alloc, &fewer));
}

test "the grid, the merge walk, and the size bounds" {
    const grid = gridFor(.{ .width = 96, .height = 64 }, max_tokens);
    try std.testing.expectEqual(Grid{ .width_patches = 8, .height_patches = 6 }, grid);
    try std.testing.expectEqual(@as(usize, 12), grid.tokens());
    const p5 = patchPosition(5, grid);
    try std.testing.expectEqual(@as(u32, 3), p5.x);
    try std.testing.expectEqual(@as(u32, 0), p5.y);
    const p10 = patchPosition(10, grid);
    try std.testing.expectEqual(@as(u32, 4), p10.x);
    try std.testing.expectEqual(@as(u32, 1), p10.y);
    const huge = gridFor(.{ .width = 4000, .height = 3000 }, max_tokens);
    try std.testing.expect(huge.patches() <= max_patches);
    try std.testing.expectEqual(@as(usize, 1024 * 4), max_patches);
    // A lowered cap scales the image down on the same grid.
    const capped = gridFor(.{ .width = 4000, .height = 3000 }, 256);
    try std.testing.expect(capped.tokens() <= 256 and capped.tokens() > 200);
}

test "the 2-D rope table turns pairs by y then x" {
    const grid: Grid = .{ .width_patches = 2, .height_patches = 2 };
    var table: [4 * rope_pairs * 2]f32 = undefined;
    try ropeTable(&table, grid);
    // Patch 0 at (0, 0): identity everywhere.
    try std.testing.expectEqual(@as(f32, 1), table[0]);
    // Patch 1 at (1, 0): pair 0 (y = 0) identity; pair 18 (x = 1) rotated by 1 rad.
    try std.testing.expectEqual(@as(f32, 1), table[(rope_pairs + 0) * 2]);
    try std.testing.expectApproxEqAbs(@as(f32, @cos(1.0)), table[(rope_pairs + 18) * 2], 1e-6);
    // Patch 2 at (0, 1): pair 0 rotated by 1 rad, pair 18 identity.
    try std.testing.expectApproxEqAbs(@as(f32, @sin(1.0)), table[(2 * rope_pairs + 0) * 2 + 1], 1e-6);
    try std.testing.expectEqual(@as(f32, 0), table[(2 * rope_pairs + 18) * 2 + 1]);
}

test "layer norm matches a hand computation" {
    const input = [_]f32{ 1, 2, 3, 4 };
    var output: [4]f32 = undefined;
    layerNorm(&input, &output, &.{ 1, 1, 1, 2 }, &.{ 0, 0, 0, 1 });
    const scale = 1.0 / @sqrt(1.25 + 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(-1.5 * scale)), output[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(1.5 * scale * 2 + 1)), output[3], 1e-6);
}
