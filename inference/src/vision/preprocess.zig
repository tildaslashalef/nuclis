//! Image preprocessing for the projectors, exact to the reference's
//! pipeline (docs/reference/vision.md § Preprocessing): the smart size
//! (round to the patch·merge grid, then the pixel bounds), a
//! Pillow-compatible separable bicubic resize in 22-bit fixed point,
//! letterboxing with black, normalization by mean and standard deviation,
//! and the patch layout the projector's convolution reads. Pure: no I/O.
const std = @import("std");
const image = @import("image.zig");

pub const Size = struct { width: u32, height: u32 };

pub const SizeOptions = struct {
    /// The grid a side is rounded to: patch × merge.
    align_size: u32,
    /// Pixel bounds after rounding (0 disables either).
    min_pixels: u32,
    max_pixels: u32,
};

/// The reference's `calc_size_preserved_ratio`: both sides rounded to the
/// nearest multiple of `align_size` (at least one), then scaled down to fit
/// `max_pixels` (floor) or up to reach `min_pixels` (ceil), in F32 as the
/// reference computes it.
pub fn smartSize(size: Size, options: SizeOptions) Size {
    const f: f32 = @floatFromInt(options.align_size);
    const width: f32 = @floatFromInt(size.width);
    const height: f32 = @floatFromInt(size.height);
    var w_bar: u32 = @max(options.align_size, roundBy(width, f));
    var h_bar: u32 = @max(options.align_size, roundBy(height, f));
    if (options.max_pixels > 0 and @as(u64, h_bar) * w_bar > options.max_pixels) {
        const beta = @sqrt(height * width / @as(f32, @floatFromInt(options.max_pixels)));
        h_bar = @max(options.align_size, floorBy(height / beta, f));
        w_bar = @max(options.align_size, floorBy(width / beta, f));
    } else if (options.min_pixels > 0 and @as(u64, h_bar) * w_bar < options.min_pixels) {
        const beta = @sqrt(@as(f32, @floatFromInt(options.min_pixels)) / (height * width));
        h_bar = ceilBy(height * beta, f);
        w_bar = ceilBy(width * beta, f);
    }
    return .{ .width = w_bar, .height = h_bar };
}
fn roundBy(x: f32, f: f32) u32 {
    return @as(u32, @intFromFloat(@round(x / f))) * @as(u32, @intFromFloat(f));
}
fn ceilBy(x: f32, f: f32) u32 {
    return @as(u32, @intFromFloat(@ceil(x / f))) * @as(u32, @intFromFloat(f));
}
fn floorBy(x: f32, f: f32) u32 {
    return @as(u32, @intFromFloat(@floor(x / f))) * @as(u32, @intFromFloat(f));
}

/// Resizes `source` into a `target`-sized image, preserving the aspect
/// ratio: the larger scale that fits (sides rounded up), centred on black
/// (the reference's `PAD_CEIL`). A source already at the target is copied.
pub fn resizeLetterbox(alloc: std.mem.Allocator, source: image.Rgb8, target: Size) ![]u8 {
    if (target.width == 0 or target.height == 0 or @as(u64, target.width) * target.height > image.max_pixels) return error.InvalidShape;
    if (source.width == target.width and source.height == target.height) return alloc.dupe(u8, source.pixels);
    const scale_w = @as(f32, @floatFromInt(target.width)) / @as(f32, @floatFromInt(source.width));
    const scale_h = @as(f32, @floatFromInt(target.height)) / @as(f32, @floatFromInt(source.height));
    const scale = @min(scale_w, scale_h);
    const new_width: u32 = @min(@as(u32, @intFromFloat(@ceil(@as(f32, @floatFromInt(source.width)) * scale))), target.width);
    const new_height: u32 = @min(@as(u32, @intFromFloat(@ceil(@as(f32, @floatFromInt(source.height)) * scale))), target.height);
    const resized = try resizeBicubic(alloc, source, .{ .width = new_width, .height = new_height });
    defer alloc.free(resized);
    const out = try alloc.alloc(u8, @as(usize, target.width) * target.height * 3);
    @memset(out, 0);
    const offset_x = (target.width - new_width) / 2;
    const offset_y = (target.height - new_height) / 2;
    for (0..new_height) |y| {
        const src_row = resized[y * new_width * 3 ..][0 .. new_width * 3];
        const dst_row = out[((y + offset_y) * target.width + offset_x) * 3 ..][0 .. new_width * 3];
        @memcpy(dst_row, src_row);
    }
    return out;
}

const precision_bits = 22;
const bicubic_support = 2.0;

/// Pillow's `ImagingResample` with the bicubic filter (a = −0.5): weights
/// per output pixel normalized in F64 then rounded to 22-bit fixed point,
/// a horizontal pass into an 8-bit intermediate, then a vertical pass.
pub fn resizeBicubic(alloc: std.mem.Allocator, source: image.Rgb8, target: Size) ![]u8 {
    if (target.width == 0 or target.height == 0 or source.width == 0 or source.height == 0) return error.InvalidShape;
    if (source.width == target.width and source.height == target.height) return alloc.dupe(u8, source.pixels);
    var current = source.pixels;
    var owned: ?[]u8 = null;
    defer if (owned) |o| alloc.free(o);
    var width = source.width;
    if (target.width != source.width) {
        const kernel = try Kernel.init(alloc, source.width, target.width);
        defer kernel.deinit(alloc);
        const out = try alloc.alloc(u8, @as(usize, target.width) * source.height * 3);
        errdefer alloc.free(out);
        for (0..source.height) |y| {
            const src_row = current[y * width * 3 ..];
            const dst_row = out[y * target.width * 3 ..];
            for (0..target.width) |xx| {
                const xmin = kernel.bounds[xx * 2];
                const count = kernel.bounds[xx * 2 + 1];
                const k = kernel.weights[xx * kernel.size ..][0..count];
                var sums = [3]i32{ 1 << (precision_bits - 1), 1 << (precision_bits - 1), 1 << (precision_bits - 1) };
                for (k, 0..) |w, x| for (0..3) |c| {
                    sums[c] +%= @as(i32, src_row[(xmin + x) * 3 + c]) *% w;
                };
                for (0..3) |c| dst_row[xx * 3 + c] = clip8(sums[c] >> precision_bits);
            }
        }
        owned = out;
        current = out;
        width = target.width;
    }
    if (target.height != source.height) {
        const kernel = try Kernel.init(alloc, source.height, target.height);
        defer kernel.deinit(alloc);
        const row = @as(usize, width) * 3;
        const out = try alloc.alloc(u8, row * target.height);
        errdefer alloc.free(out);
        const acc = try alloc.alloc(i32, row);
        defer alloc.free(acc);
        for (0..target.height) |yy| {
            const ymin = kernel.bounds[yy * 2];
            const count = kernel.bounds[yy * 2 + 1];
            @memset(acc, 1 << (precision_bits - 1));
            for (0..count) |y| {
                const src_row = current[(ymin + y) * row ..][0..row];
                const w = kernel.weights[yy * kernel.size + y];
                for (acc, src_row) |*a, v| a.* +%= @as(i32, v) *% w;
            }
            for (out[yy * row ..][0..row], acc) |*d, a| d.* = clip8(a >> precision_bits);
        }
        if (owned) |o| alloc.free(o);
        owned = null;
        return out;
    }
    const result = owned.?;
    owned = null;
    return result;
}

fn clip8(v: i32) u8 {
    return @intCast(std.math.clamp(v, 0, 255));
}

/// One dimension's precomputed filter: `bounds[2i]` the first input pixel
/// of output pixel i, `bounds[2i + 1]` how many, `weights[i·size ..]` their
/// fixed-point weights.
const Kernel = struct {
    size: usize,
    bounds: []usize,
    weights: []i32,

    fn init(alloc: std.mem.Allocator, in_size: u32, out_size: u32) !Kernel {
        const scale: f64 = @as(f64, @floatFromInt(in_size)) / @as(f64, @floatFromInt(out_size));
        const filterscale: f64 = @max(scale, 1.0);
        const support = bicubic_support * filterscale;
        const size: usize = @as(usize, @intFromFloat(@ceil(support))) * 2 + 1;
        const bounds = try alloc.alloc(usize, @as(usize, out_size) * 2);
        errdefer alloc.free(bounds);
        const weights = try alloc.alloc(i32, @as(usize, out_size) * size);
        errdefer alloc.free(weights);
        const pre = try alloc.alloc(f64, size);
        defer alloc.free(pre);
        const fxp_scale: f64 = @as(f64, 1 << precision_bits);
        for (0..out_size) |xx| {
            const center = (@as(f64, @floatFromInt(xx)) + 0.5) * scale;
            const ss = 1.0 / filterscale;
            var xmin: i64 = @intFromFloat(center - support + 0.5);
            if (xmin < 0) xmin = 0;
            var xmax: i64 = @intFromFloat(center + support + 0.5);
            if (xmax > in_size) xmax = in_size;
            const count: usize = @intCast(xmax - xmin);
            var total: f64 = 0;
            for (0..count) |x| {
                const w = bicubic((@as(f64, @floatFromInt(x)) + @as(f64, @floatFromInt(xmin)) - center + 0.5) * ss);
                pre[x] = w;
                total += w;
            }
            for (0..count) |x| if (total != 0) {
                pre[x] /= total;
            };
            for (count..size) |x| pre[x] = 0;
            bounds[xx * 2] = @intCast(xmin);
            bounds[xx * 2 + 1] = count;
            for (0..size) |x| {
                const rounded = pre[x] * fxp_scale + (if (pre[x] < 0) @as(f64, -0.5) else 0.5);
                weights[xx * size + x] = @intFromFloat(rounded);
            }
        }
        return .{ .size = size, .bounds = bounds, .weights = weights };
    }
    fn deinit(self: Kernel, alloc: std.mem.Allocator) void {
        alloc.free(self.bounds);
        alloc.free(self.weights);
    }
};

fn bicubic(distance: f64) f64 {
    const a = -0.5;
    const x = @abs(distance);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

pub const PatchOptions = struct {
    patch: u32,
    merge: u32,
    mean: [3]f32,
    std: [3]f32,
    /// `normalized · scale + bias` after the mean/std step (Gemma 4's
    /// SigLIP input is `2x − 1`).
    scale: f32 = 1,
    bias: f32 = 0,
    /// Round each value to F16, as a convolution whose im2col is F16 reads
    /// it; false keeps F32 (an im2col in the input's own type).
    half: bool = true,
};

/// The patch grid of a resized image and its rows in the projector's
/// order. Row `t` is patch `(x, y)` of the `merge × merge` block walk:
/// blocks in raster order, inside a block top-left, top-right,
/// bottom-left, bottom-right; each row is `patch · patch · 3` values laid
/// out channel-planar (`c · patch² + ky · patch + kx`), normalized as
/// `(v / 255 − mean) / std` (then `options.scale`/`bias`) and rounded to
/// F16 when `options.half`, as the reference's convolution reads them.
pub const Patches = struct {
    width_patches: u32,
    height_patches: u32,
    row: usize,
    /// `width_patches · height_patches` rows of `row` values, owned.
    values: []f32,

    pub fn count(self: Patches) usize {
        return @as(usize, self.width_patches) * self.height_patches;
    }
    pub fn deinit(self: *Patches, alloc: std.mem.Allocator) void {
        alloc.free(self.values);
        self.* = undefined;
    }
};

/// Lays out an image already resized to a multiple of `patch · merge` per side.
pub fn patches(alloc: std.mem.Allocator, pixels: []const u8, size: Size, options: PatchOptions) !Patches {
    const block = options.patch * options.merge;
    if (options.patch == 0 or options.merge == 0 or size.width % block != 0 or size.height % block != 0 or size.width == 0 or size.height == 0) return error.InvalidShape;
    if (pixels.len != @as(usize, size.width) * size.height * 3) return error.InvalidShape;
    const wp = size.width / options.patch;
    const hp = size.height / options.patch;
    const row = @as(usize, options.patch) * options.patch * 3;
    const values = try alloc.alloc(f32, @as(usize, wp) * hp * row);
    errdefer alloc.free(values);
    var t: usize = 0;
    var y2: u32 = 0;
    while (y2 < hp) : (y2 += options.merge) {
        var x2: u32 = 0;
        while (x2 < wp) : (x2 += options.merge) {
            for (0..options.merge) |dy| for (0..options.merge) |dx| {
                const px = x2 + @as(u32, @intCast(dx));
                const py = y2 + @as(u32, @intCast(dy));
                const out = values[t * row ..][0..row];
                for (0..3) |c| for (0..options.patch) |ky| for (0..options.patch) |kx| {
                    const x = px * options.patch + @as(u32, @intCast(kx));
                    const y = py * options.patch + @as(u32, @intCast(ky));
                    const v: f32 = @floatFromInt(pixels[(@as(usize, y) * size.width + x) * 3 + c]);
                    const normalized = (v / 255.0 - options.mean[c]) / options.std[c] * options.scale + options.bias;
                    const value: f32 = if (options.half) @floatCast(@as(f16, @floatCast(normalized))) else normalized;
                    out[c * options.patch * options.patch + ky * options.patch + kx] = value;
                };
                t += 1;
            };
        }
    }
    return .{ .width_patches = wp, .height_patches = hp, .row = row, .values = values };
}

test "the smart size rounds to the grid, then reaches the pixel bounds" {
    const o: SizeOptions = .{ .align_size = 32, .min_pixels = 8192, .max_pixels = 4194304 };
    try std.testing.expectEqual(Size{ .width = 128, .height = 96 }, smartSize(.{ .width = 96, .height = 64 }, o));
    try std.testing.expectEqual(Size{ .width = 768, .height = 768 }, smartSize(.{ .width = 768, .height = 768 }, o));
    try std.testing.expectEqual(Size{ .width = 32, .height = 256 }, smartSize(.{ .width = 20, .height = 250 }, o));
    const big = smartSize(.{ .width = 4000, .height = 3000 }, o);
    try std.testing.expect(@as(u64, big.width) * big.height <= 4194304 and big.width % 32 == 0 and big.height % 32 == 0);
    try std.testing.expectEqual(Size{ .width = 2336, .height = 1760 }, big);
}

test "the letterboxed bicubic resize of the fixture matches the reference's pixels" {
    const alloc = std.testing.allocator;
    var source = try image.decodePpm(alloc, @embedFile("fixtures/synthetic-96x64.ppm"));
    defer source.deinit(alloc);
    var expected = try image.decodePpm(alloc, @embedFile("fixtures/synthetic-96x64-resized.ppm"));
    defer expected.deinit(alloc);
    const target = smartSize(.{ .width = source.width, .height = source.height }, .{ .align_size = 32, .min_pixels = 8192, .max_pixels = 4194304 });
    const resized = try resizeLetterbox(alloc, source, target);
    defer alloc.free(resized);
    try std.testing.expectEqual(@as(usize, 128 * 96 * 3), resized.len);
    try std.testing.expectEqualSlices(u8, expected.pixels, resized);
}

test "patches walk merge blocks and normalize channel-planar" {
    const alloc = std.testing.allocator;
    // A 4×4 image (patch 2, merge 2): one block of four patches.
    var pixels: [4 * 4 * 3]u8 = undefined;
    for (0..16) |i| for (0..3) |c| {
        pixels[i * 3 + c] = @intCast(i * 10 + c);
    };
    var p = try patches(alloc, &pixels, .{ .width = 4, .height = 4 }, .{ .patch = 2, .merge = 2, .mean = .{ 0.5, 0.5, 0.5 }, .std = .{ 0.5, 0.5, 0.5 } });
    defer p.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 4), p.count());
    try std.testing.expectEqual(@as(usize, 12), p.row);
    // Top-right patch (dx = 1, dy = 0) is row 1; its channel 1, ky 1, kx 0 is pixel (2, 1) = index 6.
    const expected: f16 = @floatCast((@as(f32, 61.0) / 255.0 - 0.5) / 0.5);
    try std.testing.expectEqual(@as(f32, @floatCast(expected)), p.values[1 * 12 + 1 * 4 + 1 * 2 + 0]);
    try std.testing.expectError(error.InvalidShape, patches(alloc, &pixels, .{ .width = 4, .height = 4 }, .{ .patch = 3, .merge = 2, .mean = .{ 0, 0, 0 }, .std = .{ 1, 1, 1 } }));
}
