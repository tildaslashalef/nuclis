//! Seeded weights in every storage layout, for generated test models
//! (`models/tiny.zig`). Test support, never reached from the application.
//!
//! A random model has no target values to fit, so a block here is sampled
//! from the format's contract rather than quantized: its code fields are
//! random and legal, and only the scale fields are set, calibrated through
//! `decode.row` so a row decodes to about `rms` around zero. Random codes
//! reach every table entry and bit plane, which a fitted block would not.
const std = @import("std");
const decode = @import("decode.zig");
const encoding = @import("../tensor/encoding.zig");

pub const Error = decode.Error;

/// Where a block keeps its f16 scale, and the second f16 that recentres an
/// unsigned code range (a `dmin` or Q5_1's `m`), as a multiple of the scale.
const Fields = struct { d: usize, offset: ?struct { at: usize, ratio: f32 } = null };

fn fields(id: u32) ?Fields {
    return switch (id) {
        2, 6, 8, 16, 17, 18, 19, 20, 21, 22, 23, 142 => .{ .d = 0 },
        7 => .{ .d = 0, .offset = .{ .at = 2, .ratio = -15.5 } },
        10 => .{ .d = 80, .offset = .{ .at = 82, .ratio = 1.5 } },
        11 => .{ .d = 108 },
        12 => .{ .d = 0, .offset = .{ .at = 2, .ratio = 7.5 } },
        13 => .{ .d = 0, .offset = .{ .at = 2, .ratio = 15.5 } },
        14 => .{ .d = 208 },
        // IQ1_M (29) spreads its scale over four words: `writeSpreadScale`.
        143 => .{ .d = 26 },
        else => null,
    };
}

fn writeHalf(bytes: []u8, value: f32) void {
    const half: f16 = @floatCast(value);
    std.mem.writeInt(u16, bytes[0..2], @bitCast(half), .little);
}

/// IQ1_M's f16 scale: one nibble in the top of each of the four scale words.
fn writeSpreadScale(target: []u8, d: f32) void {
    const half: f16 = @floatCast(d);
    const bits: u16 = @bitCast(half);
    for (0..4) |k| {
        const shift: u4 = @intCast(4 * k);
        const nibble: u16 = (bits >> shift) & 15;
        const word = std.mem.readInt(u16, target[48 + 2 * k ..][0..2], .little);
        std.mem.writeInt(u16, target[48 + 2 * k ..][0..2], (word & 0x0fff) | (nibble << 12), .little);
    }
}

/// One random block whose scale is `d`. The two restrictions are the
/// writers': Q8_0 never stores -128 and PQ2_0 never stores code 3.
fn block(id: u32, random: std.Random, d: f32, bytes: []u8) void {
    random.bytes(bytes);
    if (id == 29) return writeSpreadScale(bytes, d);
    const at = fields(id).?;
    if (id == 8) {
        for (bytes[2..]) |*code| if (code.* == 0x80) {
            code.* = 0x81;
        };
    } else if (id == 142) {
        for (bytes[2..]) |*codes| for (0..4) |k| {
            const shift: u3 = @intCast(2 * k);
            if ((codes.* >> shift) & 3 == 3) codes.* ^= @as(u8, 3 - @as(u8, @intCast(k % 3))) << shift;
        };
    }
    writeHalf(bytes[at.d..], d);
    if (at.offset) |offset| writeHalf(bytes[offset.at..], d * offset.ratio);
}

const calibration_blocks = 16;

/// The scale of a row of `id` decoding to about `rms`; for a plain layout,
/// the standard deviation itself.
fn scaleFor(id: u32, rms: f32) Error!f32 {
    const layout = encoding.layout(id) orelse return error.UnsupportedEncoding;
    if (layout.elements_per_block == 1) return rms;
    var bytes: [calibration_blocks * 256]u8 = undefined;
    var values: [calibration_blocks * 256]f32 = undefined;
    const raw = bytes[0 .. calibration_blocks * layout.bytes_per_block];
    const output = values[0 .. calibration_blocks * layout.elements_per_block];
    var prng: std.Random.DefaultPrng = .init(id);
    for (0..calibration_blocks) |b| block(id, prng.random(), 1, raw[b * layout.bytes_per_block ..][0..layout.bytes_per_block]);
    try decode.row(id, raw, output);
    var energy: f64 = 0;
    for (output) |x| energy += x * x;
    return rms / @as(f32, @floatCast(@sqrt(energy / @as(f64, @floatFromInt(output.len)))));
}

/// Fills `bytes`, whole blocks of `id`, deterministically from `random`.
pub fn row(id: u32, random: std.Random, rms: f32, bytes: []u8) Error!void {
    const layout = encoding.layout(id) orelse return error.UnsupportedEncoding;
    const width = layout.bytes_per_block;
    if (bytes.len == 0 or bytes.len % width != 0) return error.InvalidByteLength;
    if (layout.elements_per_block != 1 and id != 29 and fields(id) == null) return error.UnsupportedEncoding;
    const d = try scaleFor(id, rms);
    var offset: usize = 0;
    while (offset < bytes.len) : (offset += width) {
        const target = bytes[offset..][0..width];
        switch (id) {
            0 => std.mem.writeInt(u32, target[0..4], @bitCast(random.floatNorm(f32) * d), .little),
            1 => writeHalf(target, random.floatNorm(f32) * d),
            // BF16 is the high half of a single.
            30 => {
                const bits: u32 = @bitCast(random.floatNorm(f32) * d);
                std.mem.writeInt(u16, target[0..2], @intCast(bits >> 16), .little);
            },
            else => block(id, random, d, target),
        }
    }
}

const all_layouts = [_]u32{ 0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 30, 142, 143 };

fn sampleRow(id: u32, seed: u64, rms: f32, bytes: []u8, output: []f32) !void {
    var prng: std.Random.DefaultPrng = .init(seed);
    try row(id, prng.random(), rms, bytes);
    try decode.row(id, bytes, output);
}

test "every layout samples finite, centred rows of the requested size" {
    const gpa = std.testing.allocator;
    for (all_layouts) |id| {
        const layout = encoding.layout(id).?;
        const blocks = 4096 / layout.elements_per_block;
        const bytes = try gpa.alloc(u8, blocks * layout.bytes_per_block);
        defer gpa.free(bytes);
        const output = try gpa.alloc(f32, blocks * layout.elements_per_block);
        defer gpa.free(output);
        try sampleRow(id, 7, 0.0625, bytes, output);
        var energy: f64 = 0;
        var sum: f64 = 0;
        var distinct: usize = 0;
        for (output) |x| {
            try std.testing.expect(std.math.isFinite(x));
            energy += x * x;
            sum += x;
            if (x != output[0]) distinct += 1;
        }
        const n: f64 = @floatFromInt(output.len);
        const rms = @sqrt(energy / n);
        // Sixteen calibration blocks against a whole row: the cruder codes
        // spread by a factor of two at most.
        try std.testing.expect(rms > 0.03 and rms < 0.13);
        try std.testing.expect(@abs(sum / n) < rms / 4);
        try std.testing.expect(distinct > output.len / 2);
    }
}

test "a row is a function of its seed" {
    const gpa = std.testing.allocator;
    for (all_layouts) |id| {
        const layout = encoding.layout(id).?;
        const a = try gpa.alloc(u8, 4 * layout.bytes_per_block);
        defer gpa.free(a);
        const b = try gpa.alloc(u8, a.len);
        defer gpa.free(b);
        const c = try gpa.alloc(u8, a.len);
        defer gpa.free(c);
        const output = try gpa.alloc(f32, 4 * layout.elements_per_block);
        defer gpa.free(output);
        try sampleRow(id, 1, 0.1, a, output);
        try sampleRow(id, 1, 0.1, b, output);
        try sampleRow(id, 2, 0.1, c, output);
        try std.testing.expectEqualSlices(u8, a, b);
        try std.testing.expect(!std.mem.eql(u8, a, c));
    }
}

test "the scale lands in the field the decoder reads" {
    // Rewriting only `d` must rescale the decoded row by the same factor,
    // which pins each layout's offsets, the spread IQ1_M word included.
    for (all_layouts) |id| {
        const layout = encoding.layout(id).?;
        if (layout.elements_per_block == 1) continue;
        var bytes: [256]u8 = undefined;
        const raw = bytes[0..layout.bytes_per_block];
        var low: [256]f32 = undefined;
        var high: [256]f32 = undefined;
        const a = low[0..layout.elements_per_block];
        const b = high[0..layout.elements_per_block];
        var first: std.Random.DefaultPrng = .init(id);
        block(id, first.random(), 0.25, raw);
        try decode.row(id, raw, a);
        var second: std.Random.DefaultPrng = .init(id);
        block(id, second.random(), 0.5, raw);
        try decode.row(id, raw, b);
        for (a, b) |x, y| try std.testing.expectApproxEqAbs(2 * x, y, 1e-3 * @abs(y) + 1e-6);
    }
}

test "the writers' restrictions hold and bad requests are refused" {
    var bytes: [34 * 8]u8 = undefined;
    var prng: std.Random.DefaultPrng = .init(3);
    try row(8, prng.random(), 0.1, &bytes);
    for (0..8) |b| for (bytes[b * 34 + 2 ..][0..32]) |code| try std.testing.expect(code != 0x80);
    try row(142, prng.random(), 0.1, &bytes);
    for (0..8) |b| for (bytes[b * 34 + 2 ..][0..32]) |codes| for (0..4) |k| {
        const shift: u3 = @intCast(2 * k);
        try std.testing.expect((codes >> shift) & 3 != 3);
    };
    try std.testing.expectError(error.UnsupportedEncoding, row(999, prng.random(), 0.1, &bytes));
    try std.testing.expectError(error.InvalidByteLength, row(8, prng.random(), 0.1, bytes[0..33]));
}
