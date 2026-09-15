//! Small CPU reference decoders for little-endian GGUF rows.
//!
//! Layout inspection establishes byte ranges; this module reconstructs values.
//! It allocates nothing and retains no references. Decode bounded rows into
//! caller-owned scratch, rather than expanding a whole model to floating point.
//! Block layouts are facts of the GGML storage formats; this implementation is
//! independent and verified against pinned fixtures. The IQ4_NL value table and
//! the IQ3_S codebook are format constants; see THIRD_PARTY_NOTICES.md. Q4_0
//! shares IQ4_NL's 18-byte block and nibble order with a linear code
//! (`q - 8`) in place of the table.
const std = @import("std");
const encoding = @import("../tensor/encoding.zig");
const iq3_grid = @import("iq3-grid.zig").values;

pub const Error = error{ UnsupportedEncoding, InvalidRowLength, InvalidByteLength };

/// Decode one nonempty contiguous row. `output.len` is its element count and
/// must contain whole storage blocks; `bytes` must be exactly that row's size.
/// Both slices are borrowed only for this call and must not overlap. Unaligned
/// source bytes are supported. All validation happens before writing output.
/// IEEE infinities/NaNs are preserved or propagated, not silently sanitized;
/// successful decoding validates storage, not the fitness of model weights.
pub fn row(id: u32, bytes: []const u8, output: []f32) Error!void {
    try validateRow(id, bytes.len, output.len);
    const layout = encoding.layout(id).?;
    const elements = layout.elements_per_block;
    const block_bytes = layout.bytes_per_block;

    var offset: usize = 0;
    var out: usize = 0;
    while (offset < bytes.len) : ({
        offset += block_bytes;
        out += elements;
    }) {
        const block = bytes[offset..][0..block_bytes];
        switch (id) {
            0 => output[out] = @bitCast(std.mem.readInt(u32, block[0..4], .little)),
            1 => output[out] = half(block[0..2]),
            2 => {
                const scale = half(block[0..2]);
                // IQ4_NL's byte order (low nibbles first half-row, high nibbles
                // second) with the code itself as the value, biased by eight.
                for (block[2..], 0..) |encoded, j| {
                    output[out + j] = scale * @as(f32, @floatFromInt(@as(i32, encoded & 15) - 8));
                    output[out + j + 16] = scale * @as(f32, @floatFromInt(@as(i32, encoded >> 4) - 8));
                }
            },
            8 => {
                const scale = half(block[0..2]);
                for (block[2..], 0..) |encoded, j| {
                    const signed: i8 = @bitCast(encoded);
                    output[out + j] = scale * @as(f32, @floatFromInt(signed));
                }
            },
            11 => q3Block(block, output[out..][0..256]),
            12 => kBlock(false, block, output[out..][0..256]),
            13 => kBlock(true, block, output[out..][0..256]),
            14 => q6Block(block, output[out..][0..256]),
            20 => {
                const scale = half(block[0..2]);
                // Low nibbles are the first half-row, high nibbles the second;
                // adjacent nibbles in a byte do not represent adjacent values.
                for (block[2..], 0..) |encoded, j| {
                    output[out + j] = scale * @as(f32, @floatFromInt(iq4_values[encoded & 15]));
                    output[out + j + 16] = scale * @as(f32, @floatFromInt(iq4_values[encoded >> 4]));
                }
            },
            21 => iq3Block(block, output[out..][0..256]),
            23 => iq4XsBlock(block, output[out..][0..256]),
            else => unreachable, // The supported IDs were checked above.
        }
    }
}

/// Check storage lengths without reading or writing buffers. Numerical callers
/// can validate a whole matrix before beginning work; supported encoding and
/// block-size rules remain centralized in the quantization module.
pub fn validateRow(id: u32, byte_count: usize, element_count: usize) Error!void {
    switch (id) {
        0, 1, 2, 8, 11, 12, 13, 14, 20, 21, 23 => {},
        else => return error.UnsupportedEncoding,
    }
    const layout = encoding.layout(id).?;
    const elements = layout.elements_per_block;
    const block_bytes = layout.bytes_per_block;
    if (element_count == 0 or element_count % elements != 0) return error.InvalidRowLength;
    // Compare quotients rather than multiplying a caller-controlled length.
    if (byte_count % block_bytes != 0 or byte_count / block_bytes != element_count / elements)
        return error.InvalidByteLength;
}

/// Q4_K and Q5_K share eight affine groups of 32 values. `fifth_bit` is
/// compile-time because storage offsets differ, while their scale math agrees.
/// The caller has already validated the complete block and output sizes.
fn kBlock(comptime fifth_bit: bool, block: []const u8, output: []f32) void {
    const d = half(block[0..2]);
    const dmin = half(block[2..4]);
    const scales = block[4..16];
    const low = block[if (fifth_bit) 48 else 16..];
    for (0..8) |group| {
        // The first four groups use the low six bits of separate bytes.
        // Later groups split each coefficient across a nibble and two high bits.
        const scale = if (group < 4) scales[group] & 63 else (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4);
        const minimum = if (group < 4) scales[group + 4] & 63 else (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
        const multiplier = d * @as(f32, @floatFromInt(scale));
        const offset = dmin * @as(f32, @floatFromInt(minimum));
        for (0..32) |column| {
            const byte = low[group / 2 * 32 + column];
            var value = if (group % 2 == 0) byte & 15 else byte >> 4;
            if (fifth_bit) {
                // Each of the 32 high-bit bytes supplies one bit to each group.
                const shift: u3 = @intCast(group);
                value |= ((block[16 + column] >> shift) & 1) << 4;
            }
            output[group * 32 + column] = multiplier * @as(f32, @floatFromInt(value)) - offset;
        }
    }
}

/// Q3_K stores biased six-bit scales and an inverted sign bit for each code.
/// Index individual bytes rather than aliasing packed data as host-endian words.
fn q3Block(block: []const u8, output: []f32) void {
    const d = half(block[108..110]);
    const scales = block[96..108];
    for (0..16) |group| {
        const nibble = if (group < 8) scales[group] & 15 else scales[group - 8] >> 4;
        const scale_shift: u3 = @intCast(2 * (group / 4));
        const upper = (scales[8 + group % 4] >> scale_shift) & 3;
        const scale = @as(i16, nibble | (upper << 4)) - 32;
        const multiplier = d * @as(f32, @floatFromInt(scale));
        for (0..16) |column| {
            const i = group * 16 + column;
            const low_shift: u3 = @intCast(2 * ((i % 128) / 32));
            const low = (block[32 + (i / 128) * 32 + i % 32] >> low_shift) & 3;
            const sign_shift: u3 = @intCast(i / 32);
            const nonnegative = (block[i % 32] >> sign_shift) & 1;
            // A clear high-mask bit means subtract four, not add a high bit.
            const value = @as(i16, low) - @as(i16, if (nonnegative == 0) 4 else 0);
            output[i] = multiplier * @as(f32, @floatFromInt(value));
        }
    }
}

/// Q6_K uses signed byte scales and six-bit codes biased by 32. Within each
/// 128-value half, low nibbles pair quarters 0/2 and 1/3; high bits pack all four.
fn q6Block(block: []const u8, output: []f32) void {
    const d = half(block[208..210]);
    for (output, 0..) |*result, i| {
        const quarter = (i % 128) / 32;
        const column = i % 32;
        const low_shift: u3 = @intCast((quarter / 2) * 4);
        const high_shift: u3 = @intCast(quarter * 2);
        const low = (block[(i / 128) * 64 + (quarter % 2) * 32 + column] >> low_shift) & 15;
        const high = (block[128 + (i / 128) * 32 + column] >> high_shift) & 3;
        const value = @as(i16, low | (high << 4)) - 32;
        const scale: i8 = @bitCast(block[192 + i / 16]);
        result.* = (d * @as(f32, @floatFromInt(scale))) * @as(f32, @floatFromInt(value));
    }
}

/// IQ4_XS reuses IQ4_NL's nonlinear values, with eight biased six-bit scales.
fn iq4XsBlock(block: []const u8, output: []f32) void {
    const d = half(block[0..2]);
    const high = std.mem.readInt(u16, block[2..4], .little);
    for (0..8) |group| {
        const low_shift: u3 = @intCast(4 * (group % 2));
        const high_shift: u4 = @intCast(2 * group);
        const low = (block[4 + group / 2] >> low_shift) & 15;
        const scale = @as(i32, low | (((high >> high_shift) & 3) << 4)) - 32;
        const multiplier = d * @as(f32, @floatFromInt(scale));
        for (block[8 + group * 16 ..][0..16], 0..) |byte, j| {
            output[group * 32 + j] = multiplier * @as(f32, @floatFromInt(iq4_values[byte & 15]));
            output[group * 32 + j + 16] = multiplier * @as(f32, @floatFromInt(iq4_values[byte >> 4]));
        }
    }
}

/// IQ3_S codes select four positive magnitudes from a 512-entry grid. Signs
/// are separate per-value bits; one odd scale multiplier applies to 32 values.
fn iq3Block(block: []const u8, output: []f32) void {
    const d = half(block[0..2]);
    for (0..8) |group| {
        const scale_shift: u3 = @intCast(4 * (group % 2));
        const scale = (block[106 + group / 2] >> scale_shift) & 15;
        const multiplier = d * @as(f32, @floatFromInt(1 + 2 * scale));
        for (0..8) |entry| {
            const index_shift: u3 = @intCast(entry);
            const index = @as(u16, block[2 + group * 8 + entry]) |
                (@as(u16, (block[66 + group] >> index_shift) & 1) << 8);
            const grid = iq3_grid[index];
            for (0..4) |component| {
                const i = group * 32 + entry * 4 + component;
                // Extract numerical bytes explicitly, independent of host endian.
                const grid_shift: u5 = @intCast(8 * component);
                const magnitude = (grid >> grid_shift) & 255;
                const sign_shift: u3 = @intCast(i % 8);
                const negative = (block[74 + i / 8] >> sign_shift) & 1;
                const value = multiplier * @as(f32, @floatFromInt(magnitude));
                output[i] = if (negative != 0) -value else value;
            }
        }
    }
}

fn half(bytes: *const [2]u8) f32 {
    const value: f16 = @bitCast(std.mem.readInt(u16, bytes, .little));
    return @floatCast(value);
}

const iq4_values = [16]i8{ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

test "F32 and F16 decode unaligned little-endian storage and IEEE edge values" {
    var output: [6]f32 = undefined;
    // Deliberate leading byte: no pointer alignment or host-endian assumptions.
    const single = [_]u8{ 99, 0, 0, 0xc0, 0x3f, 0, 0, 0x10, 0xc0 };
    try row(0, single[1..], output[0..2]);
    try std.testing.expectEqualSlices(f32, &.{ 1.5, -2.25 }, output[0..2]);
    const halves = [_]u8{ 0, 0x80, 1, 0, 0, 0x3e, 0xff, 0x7b, 0, 0x7c, 0, 0x7e };
    try row(1, &halves, &output);
    try std.testing.expectEqual(@as(u32, 0x80000000), @as(u32, @bitCast(output[0])));
    try std.testing.expectEqual(@as(f32, 0x1p-24), output[1]);
    try std.testing.expectEqual(@as(f32, 1.5), output[2]);
    try std.testing.expectEqual(@as(f32, 65504), output[3]);
    try std.testing.expect(std.math.isPositiveInf(output[4]));
    try std.testing.expect(std.math.isNan(output[5]));
}

test "Q8_0 signed extremes and independent block scales" {
    var bytes = [_]u8{0} ** 68;
    bytes[1] = 0x38; // f16 0.5
    bytes[2] = 0x80;
    bytes[3] = 0xff;
    bytes[4] = 0x7f;
    bytes[35] = 0xc0; // f16 -2
    bytes[36] = 3;
    bytes[67] = 0xfe;
    var output: [64]f32 = undefined;
    try row(8, &bytes, &output);
    var expected = [_]f32{0} ** 64;
    expected[0] = -64;
    expected[1] = -0.5;
    expected[2] = 63.5;
    expected[32] = -6;
    expected[63] = 4;
    try std.testing.expectEqualSlices(f32, &expected, &output);
}

test "IQ4_NL all codes, half-row ordering, and successive scales" {
    const encoded = [_]u8{ 0, 0x38, 0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b, 0x3c, 0x2d, 0x1e, 0x0f };
    var bytes = encoded ++ encoded;
    bytes[19] = 0xbc; // second block uses -1 instead of 0.5
    const first = [_]f32{ -63.5, -52, -41.5, -32.5, -24.5, -17.5, -11, -5, 0.5, 6.5, 12.5, 19, 26.5, 34.5, 44.5, 56.5 };
    var output: [64]f32 = undefined;
    try row(20, &bytes, &output);
    try std.testing.expectEqualSlices(f32, &first, output[0..16]);
    for (first, 0..) |value, j| {
        try std.testing.expectEqual(value, output[31 - j]);
        try std.testing.expectEqual(-2 * value, output[32 + j]);
        try std.testing.expectEqual(-2 * value, output[63 - j]);
    }
}

test "Q4_0 all codes, half-row ordering, and successive scales" {
    // The same nibble bytes as the IQ4_NL test: code j in the low nibble of
    // byte j and in the high nibble of byte 15-j, so the first half-row runs
    // 0..15 and the second 15..0.
    const encoded = [_]u8{ 0, 0x38, 0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b, 0x3c, 0x2d, 0x1e, 0x0f };
    var bytes = encoded ++ encoded;
    bytes[19] = 0xbc; // second block uses -1 instead of 0.5
    var output: [64]f32 = undefined;
    try row(2, &bytes, &output);
    for (0..16) |j| {
        const code: f32 = @floatFromInt(j);
        try std.testing.expectEqual(0.5 * (code - 8), output[j]);
        try std.testing.expectEqual(0.5 * (code - 8), output[31 - j]);
        try std.testing.expectEqual(-(code - 8), output[32 + j]);
        try std.testing.expectEqual(-(code - 8), output[63 - j]);
    }
}

test "invalid encodings and row sizes leave caller output untouched" {
    var output = [_]f32{123} ** 512;
    const bytes = [_]u8{0} ** 420;
    for ([_]u32{ 0, 1, 2, 8, 11, 12, 13, 14, 20, 21, 23 }) |id| {
        const layout = encoding.layout(id).?;
        const n = layout.elements_per_block;
        const size = layout.bytes_per_block;
        try std.testing.expectError(error.InvalidRowLength, row(id, &.{}, output[0..0]));
        try std.testing.expectError(error.InvalidByteLength, row(id, bytes[0 .. size - 1], output[0..n]));
        try std.testing.expectError(error.InvalidByteLength, row(id, bytes[0 .. size + 1], output[0..n]));
        try std.testing.expectError(error.InvalidByteLength, row(id, bytes[0..size], output[0 .. n * 2]));
        try std.testing.expectError(error.InvalidByteLength, row(id, bytes[0 .. size * 2], output[0..n]));
        if (n > 1) try std.testing.expectError(error.InvalidRowLength, row(id, bytes[0..size], output[0 .. n - 1]));
    }
    try std.testing.expectError(error.InvalidRowLength, row(8, bytes[0..34], output[0..31]));
    try std.testing.expectError(error.InvalidRowLength, row(20, bytes[0..18], output[0..33]));
    try std.testing.expectError(error.UnsupportedEncoding, row(30, &.{}, &output)); // BF16: stored, not decoded
    try std.testing.expectError(error.UnsupportedEncoding, row(999, &.{}, &output));
    for (output) |value| try std.testing.expectEqual(@as(f32, 123), value);
}

test "packed rows match pinned llama.cpp CPU decoder fixtures exactly" {
    try checkFixture(@embedFile("fixtures/simple.json"));
    try checkFixture(@embedFile("fixtures/k-affine.json"));
    try checkFixture(@embedFile("fixtures/k-signed.json"));
    try checkFixture(@embedFile("fixtures/iq.json"));
}

test "Q3_K biased scales retain all packed bits" {
    var bytes = [_]u8{0xff} ** 110;
    @memset(bytes[32..96], 0x55); // code +1 everywhere
    @memcpy(bytes[96..108], &[_]u8{ 0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe, 0, 0x55, 0xaa, 0xff });
    @memcpy(bytes[108..110], &[_]u8{ 0, 0x3c }); // d=1
    const expected = [_]f32{ -32, -14, 4, 22, -24, -6, 12, 30, -31, -13, 5, 23, -23, -5, 13, 31 };
    var output: [256]f32 = undefined;
    try row(11, &bytes, &output);
    for (output, 0..) |value, i| try std.testing.expectEqual(expected[i / 16], value);
}

test "Q3_K inverted sign mask and two-bit planes span both halves" {
    var bytes = [_]u8{0} ** 110;
    for (bytes[0..32], 0..) |*byte, i| byte.* = if (i % 2 == 0) 0x55 else 0xaa;
    @memset(bytes[32..96], 0xe4); // successive planes encode 0,1,2,3
    bytes[109] = 0x38; // d=0.5; zero packed scales decode as -32
    var output: [256]f32 = undefined;
    try row(11, &bytes, &output);
    for (output, 0..) |value, i| {
        const low: i16 = @intCast((i / 32) % 4);
        const code = low - @as(i16, if ((i / 32) % 2 == i % 2) 0 else 4);
        try std.testing.expectEqual(@as(f32, -16) * @as(f32, @floatFromInt(code)), value);
    }
}

test "Q6_K signed scale extremes and low/high quarter order" {
    var bytes = [_]u8{0xf0} ** 210;
    @memset(bytes[128..192], 0xe4);
    const scales = [_]i8{ -128, 127, -1, 0, 1, 2, -3, 4, 5, -6, 7, -8, 9, -10, 11, -12 };
    for (scales, 0..) |scale, i| bytes[192 + i] = @bitCast(scale);
    @memcpy(bytes[208..210], &[_]u8{ 0, 0x38 }); // d=0.5
    const codes = [_]f32{ -32, -16, 15, 31 };
    var output: [256]f32 = undefined;
    try row(14, &bytes, &output);
    for (output, 0..) |value, i| {
        try std.testing.expectEqual(0.5 * @as(f32, @floatFromInt(scales[i / 16])) * codes[(i / 32) % 4], value);
    }
}

fn checkFixture(json: []const u8) !void {
    const Fixture = struct {
        revision: []const u8,
        rows: []const struct { encoding: u32, bytes: []const u8, values: []const f32 },
    };
    const alloc = std.testing.allocator;
    const fixture = try std.json.parseFromSlice(Fixture, alloc, json, .{});
    defer fixture.deinit();
    try std.testing.expectEqualStrings("7620399f58aebfd2196b74021f9581bcf7218cb9", fixture.value.revision);
    for (fixture.value.rows) |sample| {
        const output = try alloc.alloc(f32, sample.values.len);
        defer alloc.free(output);
        const unaligned = try alloc.alloc(u8, sample.bytes.len + 1);
        defer alloc.free(unaligned);
        @memcpy(unaligned[1..], sample.bytes);
        try row(sample.encoding, unaligned[1..], output);
        try std.testing.expectEqualSlices(f32, sample.values, output);
    }
}

test "IQ4_XS six-bit signed scales and nonlinear half-row order" {
    var bytes = [_]u8{0xf0} ** 136;
    @memcpy(bytes[0..8], &[_]u8{ 0, 0x38, 0xe4, 0xe4, 0x10, 0x32, 0x54, 0xfe });
    const scales = [_]f32{ -32, -15, 2, 19, -28, -11, 14, 31 };
    var output: [256]f32 = undefined;
    try row(23, &bytes, &output);
    for (output, 0..) |value, i| {
        const code: f32 = if (i % 32 < 16) -127 else 113;
        try std.testing.expectEqual(0.5 * scales[i / 32] * code, value);
    }
}

test "IQ3_S grid component order, individual signs, and odd group scales" {
    var bytes = [_]u8{0} ** 110;
    bytes[1] = 0x38; // d=0.5
    @memset(bytes[2..66], 1); // grid entry 1 is {3,1,1,1}
    for (bytes[74..106], 0..) |*byte, i| byte.* = @as(u8, 1) << @as(u3, @intCast(i % 8));
    @memcpy(bytes[106..110], &[_]u8{ 0x10, 0x32, 0x54, 0xf6 });
    const scales = [_]f32{ 1, 3, 5, 7, 9, 11, 13, 31 };
    var output: [256]f32 = undefined;
    try row(21, &bytes, &output);
    for (output, 0..) |value, i| {
        const magnitude: f32 = if (i % 4 == 0) 3 else 1;
        const sign: f32 = if (i % 8 == (i / 8) % 8) -1 else 1;
        try std.testing.expectEqual(0.5 * scales[i / 32] * magnitude * sign, value);
    }
}

test "Q4_K six-bit coefficients and affine offsets for every group" {
    var bytes = [_]u8{0xa3} ** 144;
    @memcpy(bytes[0..4], &[_]u8{ 0, 0x38, 0, 0x34 }); // d=0.5, dmin=0.25
    @memcpy(bytes[4..16], &[_]u8{ 0xc1, 0x82, 0x43, 4, 0x45, 0x86, 0xc7, 8, 0x91, 0xa2, 0xb3, 0xc4 });
    // Independently unpacked coefficients: scales 1,2,3,4,49,34,19,4;
    // minima 5,6,7,8,25,42,59,12. Low/high nibbles encode 3/10.
    const expected = [_]f32{ 0.25, 8.5, 2.75, 18, 67.25, 159.5, 13.75, 17 };
    var output: [256]f32 = undefined;
    try row(12, &bytes, &output);
    for (output, 0..) |value, i| try std.testing.expectEqual(expected[i / 32], value);
}

test "Q5_K high-bit plane selects the correct group and column" {
    var bytes = [_]u8{0} ** 176;
    bytes[1] = 0x3c; // d=1; all eight group scales=1, minima=0
    @memcpy(bytes[4..16], &[_]u8{ 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1 });
    @memset(bytes[48..], 0xa3);
    for (0..32) |column| bytes[16 + column] = @as(u8, 1) << @as(u3, @intCast(column % 8));
    var output: [256]f32 = undefined;
    try row(13, &bytes, &output);
    for (output, 0..) |value, i| {
        const group = i / 32;
        const base: f32 = if (group % 2 == 0) 3 else 10;
        try std.testing.expectEqual(base + @as(f32, if (i % 8 == group) 16 else 0), value);
    }
}
