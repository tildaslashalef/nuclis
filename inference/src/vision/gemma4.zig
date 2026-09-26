//! Gemma 4's two projectors. The 12B's `gemma4uv` embeds 48-pixel patches
//! with norms and learned x/y tables and no attention (the language model
//! does the vision work); `gemma4v` is a SigLIP encoder with 2-D NEOX RoPE
//! and a 3×3 average pool, 27 blocks on the 26B-A4B (with a
//! standardization) and 16 narrower blocks on the E4B (whose matrices clamp
//! their inputs and outputs, `Clamp`).
//! Both place one token per 48×48 pixels, raster order. `bind` validates the
//! companion file; `Runtime` is the CPU reference (F64 accumulation), and
//! the host-side tables are shared with the Metal plan
//! (`gemma4_metal.zig`). Facts and traces: docs/reference/vision.md
//! § Gemma 4's projectors.
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const weights = @import("../runtime/weights.zig");
const cpu = @import("../backends/cpu/root.zig");
const preprocess = @import("preprocess.zig");
const qwen3vl = @import("qwen3vl.zig");
const inventory = @import("../models/inventory.zig");

const Tensor = gguf.Tensor;
const Binder = qwen3vl.Binder;
pub const Norm = qwen3vl.Norm;
pub const Error = qwen3vl.Error;
/// The Metal plan, the same schedules on the device.
pub const Plan = @import("gemma4_metal.zig").Plan;

pub const Kind = enum {
    /// `gemma4uv`: the unified embedder of the dense 12B.
    unified,
    /// `gemma4v`: the SigLIP encoder of the 26B-A4B.
    siglip,
};

/// Pixels per token side: the unified patch, or 16-pixel patches pooled 3×3.
pub const token_side = 48;
pub const token_pixels = token_side * token_side;
/// The reference's token bounds (`set_limit_image_tokens(70, 1120)`).
pub const min_tokens = 70;
pub const max_tokens = 1120;
pub const rms_epsilon: f32 = 1e-6;

pub const unified = struct {
    pub const patch = token_side;
    pub const values = patch * patch * 3;
    pub const hidden = 3840;
    /// Entries per axis of the x and y position tables.
    pub const table = 1120;
    /// The PyTorch LayerNorm default the embedder uses (not the file's 1e-6).
    pub const layer_norm_epsilon: f32 = 1e-5;
};

pub const siglip = struct {
    pub const patch = 16;
    pub const pool = 3;
    pub const values = patch * patch * 3;
    pub const table = 10240;
    pub const rope_base: f64 = 100;
    pub const max_patches = max_tokens * pool * pool;
    /// The widest encoder of the pinned files; workspaces size by these.
    pub const max_blocks = 27;
    pub const max_hidden = 1152;

    /// One pinned encoder's dimensions, selected by `clip.vision.block_count`.
    pub const Geometry = struct {
        hidden: usize,
        ffn: usize,
        heads: usize,
        blocks: usize,

        pub fn headDim(self: Geometry) usize {
            return self.hidden / self.heads;
        }
        /// Rotation pairs per half head: the first half of a head turns
        /// with the patch's x, the second with its y.
        pub fn ropePairs(self: Geometry) usize {
            return self.headDim() / 4;
        }
    };
    /// The 26B-A4B's encoder: head 72, 18 pairs per half.
    pub const large: Geometry = .{ .hidden = 1152, .ffn = 4304, .heads = 16, .blocks = 27 };
    /// The E4B's encoder: head 64, 16 pairs per half.
    pub const small: Geometry = .{ .hidden = 768, .ffn = 3072, .heads = 12, .blocks = 16 };
    pub const geometries = [_]*const Geometry{ &large, &small };
};

/// The image in tokens (a 48-pixel grid) and, for the SigLIP encoder, in
/// 16-pixel patches.
pub const Grid = struct {
    width_tokens: u32,
    height_tokens: u32,
    pub fn tokens(self: Grid) usize {
        return @as(usize, self.width_tokens) * self.height_tokens;
    }
    pub fn pixels(self: Grid) preprocess.Size {
        return .{ .width = self.width_tokens * token_side, .height = self.height_tokens * token_side };
    }
    pub fn widthPatches(self: Grid) u32 {
        return self.width_tokens * siglip.pool;
    }
    pub fn heightPatches(self: Grid) u32 {
        return self.height_tokens * siglip.pool;
    }
};

/// The token grid of an image: the reference's smart size on the 48-pixel
/// grid within `[min, max]` tokens. `min` is the reference's 70 except
/// where a check pins a smaller fixture; `max` is at most `max_tokens`.
pub fn gridFor(size: preprocess.Size, min: u32, max: u32) Grid {
    const target = preprocess.smartSize(size, .{ .align_size = token_side, .min_pixels = min * token_pixels, .max_pixels = max * token_pixels });
    return .{ .width_tokens = target.width / token_side, .height_tokens = target.height / token_side };
}

/// The patch rows each projector reads from an image already resized to
/// `grid.pixels()`: F32 48-pixel patches for the unified embedder, F16
/// 16-pixel patches of `2x − 1` for SigLIP (its convolution's im2col).
pub fn patchOptions(kind: Kind, mean: [3]f32, std_dev: [3]f32) preprocess.PatchOptions {
    return switch (kind) {
        .unified => .{ .patch = unified.patch, .merge = 1, .mean = mean, .std = std_dev, .half = false },
        .siglip => .{ .patch = siglip.patch, .merge = 1, .mean = mean, .std = std_dev, .scale = 2, .bias = -1 },
    };
}

pub const UnifiedBinding = struct {
    patch_norm_1: Norm,
    /// F32 `[hidden × values]`.
    patch_embedding: *const Tensor,
    patch_bias: *const Tensor,
    patch_norm_2: Norm,
    /// F32 `[2 · table][hidden]`: the x table, then the y table.
    position: *const Tensor,
    patch_norm_3: Norm,
    projection: *const Tensor,
};

/// The matrices of a block that may clamp (`Clamp`), in `Block.clamps` order.
pub const Linear = enum { query, key, value, output, gate, up, down };

/// A clipped linear's F32 scalar bounds (`<matrix>.input_min` and so on):
/// `y = clamp(W · clamp(x, input), output)`; an absent scalar leaves that
/// side open, as the reference defaults it to ±FLT_MAX.
pub const Clamp = struct {
    input_min: ?*const Tensor = null,
    input_max: ?*const Tensor = null,
    output_min: ?*const Tensor = null,
    output_max: ?*const Tensor = null,
};
/// A clamp's bounds read from the file.
pub const Bounds = struct {
    input: [2]f32,
    output: [2]f32,

    pub const open: Bounds = .{ .input = .{ -std.math.floatMax(f32), std.math.floatMax(f32) }, .output = .{ -std.math.floatMax(f32), std.math.floatMax(f32) } };
    pub fn isOpen(self: Bounds) bool {
        return std.meta.eql(self, open);
    }
};
pub fn clampBounds(view: weights.View, clamp: Clamp) !Bounds {
    var bounds = Bounds.open;
    if (clamp.input_min) |t| bounds.input[0] = try view.scalar(t, 0);
    if (clamp.input_max) |t| bounds.input[1] = try view.scalar(t, 0);
    if (clamp.output_min) |t| bounds.output[0] = try view.scalar(t, 0);
    if (clamp.output_max) |t| bounds.output[1] = try view.scalar(t, 0);
    for ([_][2]f32{ bounds.input, bounds.output }) |pair| if (std.math.isNan(pair[0]) or std.math.isNan(pair[1]) or pair[0] > pair[1]) return error.InvalidShape;
    return bounds;
}

pub const Block = struct {
    ln1: *const Tensor,
    query: *const Tensor,
    key: *const Tensor,
    value: *const Tensor,
    output: *const Tensor,
    query_norm: *const Tensor,
    key_norm: *const Tensor,
    post_attention_norm: *const Tensor,
    ln2: *const Tensor,
    gate: *const Tensor,
    up: *const Tensor,
    down: *const Tensor,
    post_ffn_norm: *const Tensor,
    clamps: [@typeInfo(Linear).@"enum".fields.len]Clamp,

    pub fn matrix(self: Block, which: Linear) *const Tensor {
        return switch (which) {
            .query => self.query,
            .key => self.key,
            .value => self.value,
            .output => self.output,
            .gate => self.gate,
            .up => self.up,
            .down => self.down,
        };
    }
};

pub const SiglipBinding = struct {
    geometry: *const siglip.Geometry,
    /// F32 `[hidden][3][patch][patch]`: a `[hidden × values]` matrix in the
    /// channel-planar column order `preprocess.patches` writes.
    patch_embedding: *const Tensor,
    position: *const Tensor,
    /// `geometry.blocks` leading entries are bound; see `active`.
    layers: [siglip.max_blocks]Block,
    /// The pooled rows' standardization, absent on the E4B's file.
    std_bias: ?*const Tensor,
    std_scale: ?*const Tensor,
    projection: *const Tensor,

    pub fn active(self: *const SiglipBinding) []const Block {
        return self.layers[0..self.geometry.blocks];
    }
};

pub const Binding = struct {
    kind: Kind,
    net: union(Kind) { unified: UnifiedBinding, siglip: SiglipBinding },
    /// The language model width the projection writes.
    output_width: usize,
    mean: [3]f32,
    std: [3]f32,
    tensors: u32,
    bytes: u64,
};

/// The projector type of a Gemma 4 companion file, or null for another.
pub fn kindOf(doc: *const gguf.Document) ?Kind {
    const value = doc.get("clip.vision.projector_type") orelse return null;
    const name = switch (value) {
        .string => |s| s,
        else => return null,
    };
    if (std.mem.eql(u8, name, "gemma4uv")) return .unified;
    if (std.mem.eql(u8, name, "gemma4v")) return .siglip;
    return null;
}

fn expectUnsigned(doc: *const gguf.Document, key: []const u8, value: u64) Error!void {
    if (try qwen3vl.unsignedValue(doc, key) != value) return error.UnsupportedConfiguration;
}

/// Whether a tensor belongs to the audio embedder that shares the file
/// (the 12B's `mm.a.*` projection, the E4B's whole `a.*` encoder).
fn isAudio(name: []const u8) bool {
    return std.mem.startsWith(u8, name, "mm.a.") or std.mem.startsWith(u8, name, "a.");
}

/// A matrix's optional clamp scalars, each an F32 `[1]` if present.
fn takeClamp(binder: *Binder, block: usize, name: []const u8) Error!Clamp {
    var result: Clamp = .{};
    inline for (.{ "input_min", "input_max", "output_min", "output_max" }) |field| {
        var buffer: [96]u8 = undefined;
        const key = std.fmt.bufPrint(&buffer, "v.blk.{d}.{s}.{s}", .{ block, name, field }) catch unreachable;
        if (binder.remaining.contains(key)) @field(result, field) = try binder.take(key, &.{1}, true);
    }
    return result;
}

/// doc must come from successful GGUF parsing; the returned tensor
/// references borrow it. The audio embedder's tensors (`isAudio`) are left
/// unbound. Allocations are lookup storage freed before return.
pub fn bind(alloc: std.mem.Allocator, doc: *const gguf.Document) Error!Binding {
    if (!std.mem.eql(u8, try qwen3vl.stringValue(doc, "general.architecture"), "clip")) return error.UnsupportedArchitecture;
    const kind = kindOf(doc) orelse return error.UnsupportedProjector;
    switch (doc.get("clip.vision.attention.layer_norm_epsilon") orelse return error.MissingMetadata) {
        .float => |f| if (@abs(f - 1e-6) > 1e-9) return error.UnsupportedConfiguration,
        else => return error.InvalidMetadata,
    }
    try expectUnsigned(doc, "clip.vision.patch_size", siglip.patch);
    var binder: Binder = .{ .remaining = .init(alloc) };
    defer binder.remaining.deinit();
    for (doc.tensors) |*tensor| {
        const slot = try binder.remaining.getOrPut(tensor.name);
        if (slot.found_existing) return error.DuplicateTensor;
        slot.value_ptr.* = tensor;
    }
    var result: Binding = undefined;
    result.kind = kind;
    result.mean = try qwen3vl.floatTriple(doc, "clip.vision.image_mean");
    result.std = try qwen3vl.floatTriple(doc, "clip.vision.image_std");
    for (result.std) |s| if (!(s > 0)) return error.UnsupportedConfiguration;
    const width = try qwen3vl.unsignedValue(doc, "clip.vision.projection_dim");
    if (width == 0 or width > 16384) return error.UnsupportedConfiguration;
    result.output_width = @intCast(width);
    switch (kind) {
        .unified => {
            const h = unified.hidden;
            try expectUnsigned(doc, "clip.vision.embedding_length", h);
            try expectUnsigned(doc, "clip.vision.block_count", 0);
            result.net = .{ .unified = .{
                .patch_norm_1 = try binder.norm("v.patch_norm.1", .{}, unified.values),
                .patch_embedding = try binder.take("v.patch_embd.weight", &.{ unified.values, h }, true),
                .patch_bias = try binder.take("v.patch_embd.bias", &.{h}, true),
                .patch_norm_2 = try binder.norm("v.patch_norm.2", .{}, h),
                .position = try binder.take("v.position_embd.weight", &.{ h, unified.table, 2 }, true),
                .patch_norm_3 = try binder.norm("v.patch_norm.3", .{}, h),
                .projection = try binder.take("mm.input_projection.weight", &.{ h, width }, false),
            } };
            // The audio embedder shares the file; its tensors are not ours.
            var it = binder.remaining.keyIterator();
            while (it.next()) |name| if (!isAudio(name.*)) return error.UnexpectedTensor;
        },
        .siglip => {
            const blocks = try qwen3vl.unsignedValue(doc, "clip.vision.block_count");
            const geometry = for (siglip.geometries) |g| {
                if (g.blocks == blocks) break g;
            } else return error.UnsupportedConfiguration;
            const h = geometry.hidden;
            const ffn = geometry.ffn;
            try expectUnsigned(doc, "clip.vision.embedding_length", h);
            try expectUnsigned(doc, "clip.vision.feed_forward_length", ffn);
            try expectUnsigned(doc, "clip.vision.attention.head_count", geometry.heads);
            // The FFN gate is the reference's default for a file without
            // `clip.use_gelu`/`use_silu` (gelu_quick); a file naming one is not this graph.
            if (doc.get("clip.use_gelu") != null or doc.get("clip.use_silu") != null) return error.UnsupportedConfiguration;
            var net: SiglipBinding = undefined;
            net.geometry = geometry;
            net.patch_embedding = try binder.take("v.patch_embd.weight", &.{ siglip.patch, siglip.patch, 3, h }, true);
            net.position = try binder.take("v.position_embd.weight", &.{ h, siglip.table, 2 }, true);
            const d = geometry.headDim();
            for (net.layers[0..geometry.blocks], 0..) |*layer, i| layer.* = .{
                .ln1 = try binder.named("v.blk.{d}.ln1.weight", .{i}, &.{h}, true),
                .query = try binder.named("v.blk.{d}.attn_q.weight", .{i}, &.{ h, h }, false),
                .key = try binder.named("v.blk.{d}.attn_k.weight", .{i}, &.{ h, h }, false),
                .value = try binder.named("v.blk.{d}.attn_v.weight", .{i}, &.{ h, h }, false),
                .output = try binder.named("v.blk.{d}.attn_out.weight", .{i}, &.{ h, h }, false),
                .query_norm = try binder.named("v.blk.{d}.attn_q_norm.weight", .{i}, &.{d}, true),
                .key_norm = try binder.named("v.blk.{d}.attn_k_norm.weight", .{i}, &.{d}, true),
                .post_attention_norm = try binder.named("v.blk.{d}.attn_post_norm.weight", .{i}, &.{h}, true),
                .ln2 = try binder.named("v.blk.{d}.ln2.weight", .{i}, &.{h}, true),
                .gate = try binder.named("v.blk.{d}.ffn_gate.weight", .{i}, &.{ h, ffn }, false),
                .up = try binder.named("v.blk.{d}.ffn_up.weight", .{i}, &.{ h, ffn }, false),
                .down = try binder.named("v.blk.{d}.ffn_down.weight", .{i}, &.{ ffn, h }, false),
                .post_ffn_norm = try binder.named("v.blk.{d}.ffn_post_norm.weight", .{i}, &.{h}, true),
                .clamps = .{
                    try takeClamp(&binder, i, "attn_q"),
                    try takeClamp(&binder, i, "attn_k"),
                    try takeClamp(&binder, i, "attn_v"),
                    try takeClamp(&binder, i, "attn_out"),
                    try takeClamp(&binder, i, "ffn_gate"),
                    try takeClamp(&binder, i, "ffn_up"),
                    try takeClamp(&binder, i, "ffn_down"),
                },
            };
            // The standardization comes as a pair or not at all.
            const has_std = binder.remaining.contains("v.std_bias");
            net.std_bias = if (has_std) try binder.take("v.std_bias", &.{h}, true) else null;
            net.std_scale = if (has_std) try binder.take("v.std_scale", &.{h}, true) else null;
            net.projection = try binder.take("mm.input_projection.weight", &.{ h, width }, false);
            var it = binder.remaining.keyIterator();
            while (it.next()) |name| if (!isAudio(name.*)) return error.UnexpectedTensor;
            result.net = .{ .siglip = net };
        },
    }
    result.tensors = binder.tensors;
    result.bytes = binder.bytes;
    return result;
}

/// Row `index` of an F32 table stored as `[rows][width]`.
pub fn tableRow(view: weights.View, tensor: *const Tensor, width: usize, index: usize, out: []f32) !void {
    const raw = try view.bytes(tensor);
    if (out.len != width or (index + 1) * width * 4 > raw.len) return error.InvalidShape;
    for (out, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, raw[(index * width + i) * 4 ..][0..4], .little));
}

/// The learned position rows of a raster grid `columns` wide: the x table
/// at `i % columns` plus the y table at `i / columns`, summed in F32 as the
/// reference's two adds. `out` holds `count · width`; `scratch` one row.
pub fn positionRows(view: weights.View, tensor: *const Tensor, width: usize, table: usize, columns: usize, count: usize, out: []f32, scratch: []f32) !void {
    if (columns == 0 or out.len < count * width or scratch.len < width) return error.InvalidShape;
    for (0..count) |i| {
        const x = i % columns;
        const y = i / columns;
        if (x >= table or y >= table) return error.InvalidShape;
        const row = out[i * width ..][0..width];
        try tableRow(view, tensor, width, x, row);
        try tableRow(view, tensor, width, table + y, scratch[0..width]);
        for (row, scratch[0..width]) |*r, s| r.* += s;
    }
}

/// The SigLIP 2-D rotary tables for a patch grid, raster order: per patch
/// `pairs` (cos, sin) pairs of its x (`x_table`) and of its y (`y_table`) at
/// frequencies `100^(−2j/(2·pairs))`, F64 angles rounded to F32.
pub fn ropeTables(x_table: []f32, y_table: []f32, pairs: usize, width_patches: usize, count: usize) !void {
    if (width_patches == 0 or x_table.len < count * pairs * 2 or y_table.len < count * pairs * 2) return error.InvalidShape;
    for (0..count) |i| {
        const positions = [2]f64{ @floatFromInt(i % width_patches), @floatFromInt(i / width_patches) };
        for ([2][]f32{ x_table, y_table }, positions) |table, position| for (0..pairs) |j| {
            const theta = position * std.math.pow(f64, siglip.rope_base, -2.0 * @as(f64, @floatFromInt(j)) / @as(f64, @floatFromInt(2 * pairs)));
            table[(i * pairs + j) * 2] = @floatCast(@cos(theta));
            table[(i * pairs + j) * 2 + 1] = @floatCast(@sin(theta));
        };
    }
}

/// The SigLIP patch kernel as a `[hidden × values]` F32 matrix view of the
/// file's bytes (GGUF `[kx, ky, c, o]`, so each output row is already
/// channel-planar).
pub fn siglipKernel(view: weights.View, net: *const SiglipBinding) !cpu.Matrix {
    return .{ .encoding = 0, .rows = net.geometry.hidden, .columns = siglip.values, .bytes = try view.bytes(net.patch_embedding) };
}

/// `rms(input)·weight` (or without a weight), statistics in F64.
fn rmsNorm(input: []const f32, output: []f32, weight: ?[]const f32) !void {
    try cpu.rmsNorm(input, output, rms_epsilon);
    if (weight) |w| for (output, w) |*o, x| {
        o.* *= x;
    };
}

/// Rotates one half head NEOX-style: pairs `(j, j + pairs)`.
fn rotateHalf(half: []f32, table: []const f32) void {
    const pairs = half.len / 2;
    for (0..pairs) |j| {
        const c: f64 = table[j * 2];
        const s: f64 = table[j * 2 + 1];
        const a: f64 = half[j];
        const b: f64 = half[j + pairs];
        half[j] = @floatCast(a * c - b * s);
        half[j + pairs] = @floatCast(a * s + b * c);
    }
}

/// The CPU reference for either projector: every product through
/// `cpu.matvec` (F64 accumulation), norms and attention in F64.
pub const Runtime = struct {
    alloc: std.mem.Allocator,
    view: weights.View,
    binding: *const Binding,
    /// The small F32 vectors one `encode` decoded, freed when it returns.
    vectors: std.ArrayList([]f32),

    pub fn init(alloc: std.mem.Allocator, view: weights.View, binding: *const Binding) !Runtime {
        return .{ .alloc = alloc, .view = view, .binding = binding, .vectors = .empty };
    }
    pub fn deinit(self: *Runtime) void {
        self.releaseVectors();
        self.vectors.deinit(self.alloc);
        self.* = undefined;
    }
    fn releaseVectors(self: *Runtime) void {
        for (self.vectors.items) |v| self.alloc.free(v);
        self.vectors.clearRetainingCapacity();
    }
    fn vector(self: *Runtime, tensor: *const Tensor) ![]f32 {
        const out = try self.view.vector(self.alloc, tensor);
        errdefer self.alloc.free(out);
        try self.vectors.append(self.alloc, out);
        return out;
    }

    /// Encodes `patches` (from `preprocess.patches` with `patchOptions`)
    /// of an image of `grid` into `out`: `grid.tokens() × output_width`.
    pub fn encode(self: *Runtime, patches: preprocess.Patches, grid: Grid, out: []f32) !void {
        if (grid.tokens() == 0 or grid.tokens() > max_tokens or out.len < grid.tokens() * self.binding.output_width) return error.InvalidShape;
        defer self.releaseVectors();
        switch (self.binding.net) {
            .unified => |*net| try self.encodeUnified(net, patches, grid, out),
            .siglip => |*net| try self.encodeSiglip(net, patches, grid, out),
        }
    }

    fn encodeUnified(self: *Runtime, net: *const UnifiedBinding, patches: preprocess.Patches, grid: Grid, out: []f32) !void {
        const n = grid.tokens();
        const h = unified.hidden;
        const width = self.binding.output_width;
        if (patches.row != unified.values or patches.count() != n or patches.width_patches != grid.width_tokens) return error.InvalidShape;
        const a = self.alloc;
        const eps = unified.layer_norm_epsilon;
        const n1w = try self.vector(net.patch_norm_1.weight);
        const n1b = try self.vector(net.patch_norm_1.bias);
        const bias = try self.vector(net.patch_bias);
        const n2w = try self.vector(net.patch_norm_2.weight);
        const n2b = try self.vector(net.patch_norm_2.bias);
        const n3w = try self.vector(net.patch_norm_3.weight);
        const n3b = try self.vector(net.patch_norm_3.bias);
        const normed = try a.alloc(f32, unified.values);
        defer a.free(normed);
        const x = try a.alloc(f32, h);
        defer a.free(x);
        const y = try a.alloc(f32, h);
        defer a.free(y);
        const positions = try a.alloc(f32, n * h);
        defer a.free(positions);
        const scratch = try a.alloc(f32, unified.values + h);
        defer a.free(scratch);
        try positionRows(self.view, net.position, h, unified.table, grid.width_tokens, n, positions, scratch);
        const kernel = try self.view.matrix(net.patch_embedding);
        const projection = try self.view.matrix(net.projection);
        for (0..n) |t| {
            qwen3vl.layerNormEpsilon(patches.values[t * unified.values ..][0..unified.values], normed, n1w, n1b, eps);
            try cpu.matvec(kernel, normed, x, scratch);
            for (x, bias) |*v, b| v.* += b;
            qwen3vl.layerNormEpsilon(x, y, n2w, n2b, eps);
            for (y, positions[t * h ..][0..h]) |*v, p| v.* += p;
            qwen3vl.layerNormEpsilon(y, x, n3w, n3b, eps);
            try rmsNorm(x, y, null);
            try cpu.matvec(projection, y, out[t * width ..][0..width], scratch);
        }
    }

    /// `output = clamp(W · clamp(input))` (`Clamp`); `input` is left as it was.
    fn clipped(self: *Runtime, matrix: cpu.Matrix, bounds: Bounds, input: []const f32, output: []f32, scratch: []f32, staged: []f32) !void {
        _ = self;
        var source = input;
        if (!bounds.isOpen()) {
            for (staged[0..input.len], input) |*s, x| s.* = std.math.clamp(x, bounds.input[0], bounds.input[1]);
            source = staged[0..input.len];
        }
        try cpu.matvec(matrix, source, output, scratch);
        if (!bounds.isOpen()) for (output) |*y| {
            y.* = std.math.clamp(y.*, bounds.output[0], bounds.output[1]);
        };
    }

    fn encodeSiglip(self: *Runtime, net: *const SiglipBinding, patches: preprocess.Patches, grid: Grid, out: []f32) !void {
        const wp: usize = grid.widthPatches();
        const hp: usize = grid.heightPatches();
        const n = wp * hp;
        const g = net.geometry;
        const h = g.hidden;
        const d = g.headDim();
        const pairs = g.ropePairs();
        const width = self.binding.output_width;
        if (patches.row != siglip.values or patches.width_patches != wp or patches.height_patches != hp or patches.count() != n) return error.InvalidShape;
        const a = self.alloc;
        const x = try a.alloc(f32, n * h);
        defer a.free(x);
        const normed = try a.alloc(f32, n * h);
        defer a.free(normed);
        const q = try a.alloc(f32, n * h);
        defer a.free(q);
        const k = try a.alloc(f32, n * h);
        defer a.free(k);
        const v = try a.alloc(f32, n * h);
        defer a.free(v);
        const attn = try a.alloc(f32, n * h);
        defer a.free(attn);
        const gate = try a.alloc(f32, g.ffn);
        defer a.free(gate);
        const up = try a.alloc(f32, g.ffn);
        defer a.free(up);
        const row = try a.alloc(f32, h);
        defer a.free(row);
        const x_table = try a.alloc(f32, n * pairs * 2);
        defer a.free(x_table);
        const y_table = try a.alloc(f32, n * pairs * 2);
        defer a.free(y_table);
        const scratch = try a.alloc(f32, g.ffn + h);
        defer a.free(scratch);
        // A clamped copy of a matrix's input.
        const staged = try a.alloc(f32, g.ffn + h);
        defer a.free(staged);
        const scores = try a.alloc(f64, n);
        defer a.free(scores);
        try ropeTables(x_table, y_table, pairs, wp, n);

        const kernel = try siglipKernel(self.view, net);
        try positionRows(self.view, net.position, h, siglip.table, wp, n, normed, scratch);
        for (0..n) |t| {
            const xt = x[t * h ..][0..h];
            try cpu.matvec(kernel, patches.values[t * siglip.values ..][0..siglip.values], xt, scratch);
            for (xt, normed[t * h ..][0..h]) |*o, p| o.* += p;
        }
        for (net.active()) |layer| {
            var bounds: [layer.clamps.len]Bounds = undefined;
            for (&bounds, layer.clamps) |*b, c| b.* = try clampBounds(self.view, c);
            const ln1 = try self.vector(layer.ln1);
            const qn = try self.vector(layer.query_norm);
            const kn = try self.vector(layer.key_norm);
            const post_attention = try self.vector(layer.post_attention_norm);
            const ln2 = try self.vector(layer.ln2);
            const post_ffn = try self.vector(layer.post_ffn_norm);
            const wq = try self.view.matrix(layer.query);
            const wk = try self.view.matrix(layer.key);
            const wv = try self.view.matrix(layer.value);
            for (0..n) |t| {
                const nt = normed[t * h ..][0..h];
                try rmsNorm(x[t * h ..][0..h], nt, ln1);
                try self.clipped(wq, bounds[@intFromEnum(Linear.query)], nt, q[t * h ..][0..h], scratch, staged);
                try self.clipped(wk, bounds[@intFromEnum(Linear.key)], nt, k[t * h ..][0..h], scratch, staged);
                try self.clipped(wv, bounds[@intFromEnum(Linear.value)], nt, v[t * h ..][0..h], scratch, staged);
                const xs = x_table[t * pairs * 2 ..][0 .. pairs * 2];
                const ys = y_table[t * pairs * 2 ..][0 .. pairs * 2];
                for (0..g.heads) |head| {
                    const qh = q[t * h + head * d ..][0..d];
                    const kh = k[t * h + head * d ..][0..d];
                    try rmsNorm(qh, qh, qn);
                    try rmsNorm(kh, kh, kn);
                    rotateHalf(qh[0 .. d / 2], xs);
                    rotateHalf(qh[d / 2 ..], ys);
                    rotateHalf(kh[0 .. d / 2], xs);
                    rotateHalf(kh[d / 2 ..], ys);
                    const vh = v[t * h + head * d ..][0..d];
                    try rmsNorm(vh, vh, null);
                }
            }
            attention(q, k, v, attn, n, g, scores);
            const wo = try self.view.matrix(layer.output);
            const wg = try self.view.matrix(layer.gate);
            const wu = try self.view.matrix(layer.up);
            const wd = try self.view.matrix(layer.down);
            for (0..n) |t| {
                const xt = x[t * h ..][0..h];
                try self.clipped(wo, bounds[@intFromEnum(Linear.output)], attn[t * h ..][0..h], row, scratch, staged);
                try rmsNorm(row, row, post_attention);
                for (xt, row) |*o, r| o.* += r;
                const nt = normed[t * h ..][0..h];
                try rmsNorm(xt, nt, ln2);
                try self.clipped(wg, bounds[@intFromEnum(Linear.gate)], nt, gate, scratch, staged);
                try self.clipped(wu, bounds[@intFromEnum(Linear.up)], nt, up, scratch, staged);
                for (gate, up) |*gv, u| gv.* = cpu.geluQuick(gv.*) * u;
                try self.clipped(wd, bounds[@intFromEnum(Linear.down)], gate, row, scratch, staged);
                try rmsNorm(row, row, post_ffn);
                for (xt, row) |*o, r| o.* += r;
            }
        }
        const std_bias = if (net.std_bias) |t| try self.vector(t) else null;
        const std_scale = if (net.std_scale) |t| try self.vector(t) else null;
        const projection = try self.view.matrix(net.projection);
        try poolRows(x, grid, h, normed);
        for (0..grid.tokens()) |t| {
            const pooled = normed[t * h ..][0..h];
            if (std_bias) |bias| standardize(pooled, bias, std_scale.?);
            try rmsNorm(pooled, row, null);
            try cpu.matvec(projection, row, out[t * width ..][0..width], scratch);
        }
    }
};

/// The 3×3 average pool of raster patch rows of width `h` into raster
/// token rows, scaled by √h: `out` gets `grid.tokens() × h`.
pub fn poolRows(x: []const f32, grid: Grid, h: usize, out: []f32) !void {
    const wp: usize = grid.widthPatches();
    if (x.len < wp * grid.heightPatches() * h or out.len < grid.tokens() * h) return error.InvalidShape;
    const root: f32 = @sqrt(@as(f32, @floatFromInt(h)));
    for (0..grid.height_tokens) |ty| for (0..grid.width_tokens) |tx| {
        const dst = out[(ty * grid.width_tokens + tx) * h ..][0..h];
        for (dst, 0..) |*o, c| {
            var sum: f64 = 0;
            for (0..siglip.pool) |dy| for (0..siglip.pool) |dx| {
                sum += x[((ty * siglip.pool + dy) * wp + tx * siglip.pool + dx) * h + c];
            };
            // The pool's mean in F32, then the scale, as the reference's two ops.
            const mean: f32 = @floatCast(sum / (siglip.pool * siglip.pool));
            o.* = mean * root;
        }
    };
}

/// `(x − bias) ⊙ scale`, the pooled rows' standardization.
pub fn standardize(row: []f32, bias: []const f32, scale: []const f32) void {
    for (row, bias, scale) |*r, b, s| r.* = (r.* - b) * s;
}

/// Bidirectional attention of `n` rows of the geometry's heads, scale 1.
fn attention(q: []const f32, k: []const f32, v: []const f32, out: []f32, n: usize, g: *const siglip.Geometry, scores: []f64) void {
    const h = g.hidden;
    const d = g.headDim();
    for (0..g.heads) |head| for (0..n) |i| {
        const qi = q[i * h + head * d ..][0..d];
        var max: f64 = -std.math.inf(f64);
        for (0..n) |j| {
            var dot: f64 = 0;
            for (qi, k[j * h + head * d ..][0..d]) |a, b| dot += @as(f64, a) * b;
            scores[j] = dot;
            max = @max(max, dot);
        }
        var total: f64 = 0;
        for (scores[0..n]) |*s| {
            s.* = @exp(s.* - max);
            total += s.*;
        }
        const o = out[i * h + head * d ..][0..d];
        for (o, 0..) |*value, c| {
            var sum: f64 = 0;
            for (0..n) |j| sum += scores[j] * v[j * h + head * d + c];
            value.* = @floatCast(sum / total);
        }
    };
}

/// The E4B's projector inventory (`fixtures/gemma4-e4b-mmproj.json`): the
/// small SigLIP encoder with clamps, and a full audio encoder.
pub fn inventoryDocumentE4b(gpa: std.mem.Allocator) !gguf.Document {
    return inventory.document(gpa, @embedFile("fixtures/gemma4-e4b-mmproj.json"));
}

pub fn inventoryDocument(gpa: std.mem.Allocator, kind: Kind) !gguf.Document {
    return inventory.document(gpa, switch (kind) {
        .unified => @embedFile("fixtures/gemma4-12b-mmproj.json"),
        .siglip => @embedFile("fixtures/gemma4-26b-a4b-mmproj.json"),
    });
}

test "bind accepts both pinned projector inventories" {
    const alloc = std.testing.allocator;
    var uv = try inventoryDocument(alloc, .unified);
    defer uv.storage.deinit();
    const a = try bind(alloc, &uv);
    try std.testing.expectEqual(Kind.unified, a.kind);
    try std.testing.expectEqual(@as(usize, 3840), a.output_width);
    // Ten vision tensors; the audio projection stays unbound.
    try std.testing.expectEqual(@as(u32, 10), a.tensors);
    var v = try inventoryDocument(alloc, .siglip);
    defer v.storage.deinit();
    const b = try bind(alloc, &v);
    try std.testing.expectEqual(Kind.siglip, b.kind);
    try std.testing.expectEqual(@as(usize, 2816), b.output_width);
    try std.testing.expectEqual(@as(u32, 356), b.tensors);
    try std.testing.expectEqual([3]f32{ 0, 0, 0 }, b.mean);
    try std.testing.expect(b.net.siglip.std_bias != null);
    var e = try inventoryDocumentE4b(alloc);
    defer e.storage.deinit();
    const c = try bind(alloc, &e);
    try std.testing.expectEqual(Kind.siglip, c.kind);
    try std.testing.expectEqual(@as(usize, 2560), c.output_width);
    try std.testing.expectEqual(&siglip.small, c.net.siglip.geometry);
    try std.testing.expect(c.net.siglip.std_bias == null);
    // 16 blocks of 13 tensors and 28 clamp scalars, the patch kernel, the
    // position table, and the projection; the 745 audio tensors stay unbound.
    try std.testing.expectEqual(@as(u32, 16 * (13 + 28) + 3), c.tensors);
    const q = c.net.siglip.layers[0].clamps[@intFromEnum(Linear.query)];
    try std.testing.expect(q.input_min != null and q.output_max != null);
}

test "bind refuses the Qwen projector and a missing tensor" {
    const alloc = std.testing.allocator;
    var qwen = try qwen3vl.inventoryDocument(alloc);
    defer qwen.storage.deinit();
    try std.testing.expectError(error.UnsupportedProjector, bind(alloc, &qwen));
    var v = try inventoryDocument(alloc, .siglip);
    defer v.storage.deinit();
    var fewer = v;
    fewer.tensors = v.tensors[1..];
    try std.testing.expectError(error.MissingTensor, bind(alloc, &fewer));
}

test "the grid follows the reference's bounds on the 48-pixel grid" {
    // The synthetic fixture at the reference's minimum and at a pinned 4.
    try std.testing.expectEqual(Grid{ .width_tokens = 11, .height_tokens = 7 }, gridFor(.{ .width = 96, .height = 64 }, min_tokens, max_tokens));
    try std.testing.expectEqual(Grid{ .width_tokens = 3, .height_tokens = 2 }, gridFor(.{ .width = 96, .height = 64 }, 4, max_tokens));
    // The aerial photo: 3840×2160 → 44×25 tokens, under the 1,120 maximum.
    const aerial = gridFor(.{ .width = 3840, .height = 2160 }, min_tokens, max_tokens);
    try std.testing.expectEqual(Grid{ .width_tokens = 44, .height_tokens = 25 }, aerial);
    try std.testing.expect(gridFor(.{ .width = 9000, .height = 9000 }, min_tokens, max_tokens).tokens() <= max_tokens);
    const capped = gridFor(.{ .width = 3840, .height = 2160 }, min_tokens, 280);
    try std.testing.expect(capped.tokens() <= 280 and capped.tokens() > 200);
}

test "the rope tables turn the first half with x and the second with y" {
    const pairs = siglip.large.ropePairs();
    var xt: [6 * 18 * 2]f32 = undefined;
    var yt: [6 * 18 * 2]f32 = undefined;
    try ropeTables(&xt, &yt, pairs, 3, 6);
    // Patch 4 is (x 1, y 1); patch 2 is (x 2, y 0).
    try std.testing.expectApproxEqAbs(@as(f32, @cos(1.0)), xt[(4 * pairs) * 2], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, @cos(2.0)), xt[(2 * pairs) * 2], 1e-7);
    try std.testing.expectEqual(@as(f32, 1), yt[(2 * pairs) * 2]);
    // Pair 1's frequency is 100^(−2/36).
    const f = std.math.pow(f64, 100, -2.0 / 36.0);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(@sin(f))), yt[(4 * pairs + 1) * 2 + 1], 1e-7);
    // The E4B's 64-wide heads: pair 1 at 100^(−2/32).
    try std.testing.expectEqual(@as(usize, 16), siglip.small.ropePairs());
}

test "the pool averages 3×3 patch blocks in raster order and scales by √hidden" {
    const alloc = std.testing.allocator;
    const grid: Grid = .{ .width_tokens = 2, .height_tokens = 1 };
    const h = siglip.large.hidden;
    const x = try alloc.alloc(f32, 18 * h);
    defer alloc.free(x);
    for (0..18) |p| @memset(x[p * h ..][0..h], @floatFromInt(p % 6));
    const out = try alloc.alloc(f32, 2 * h);
    defer alloc.free(out);
    try poolRows(x, grid, h, out);
    // Token 0 covers patch columns 0..2 of rows 0..2: mean 1; token 1 columns 3..5: mean 4.
    try std.testing.expectApproxEqAbs(@as(f32, @sqrt(1152.0)), out[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 4 * @sqrt(1152.0)), out[h], 1e-3);
}
