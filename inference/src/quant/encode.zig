//! Reference encoder for the one block format nuclis writes at run time:
//! Q8_0, the attention cache's `q8_0` precision (runtime/session.zig).
//! Metal's `nu_pack_q8_0` computes the same arithmetic, held bit-identical
//! by `test-metal`; `decode.row` reads the blocks back.
const std = @import("std");

pub const Error = error{ InvalidRowLength, InvalidByteLength };

pub const q8_0_block = 32;
pub const q8_0_block_bytes = 34;

/// Quantizes `input` (whole 32-value blocks) into `output`
/// (`input.len / 32 * 34` bytes). Per block `d = f16(max |x| / 127)` and
/// `q = clamp(round(x / d), -127, 127)`, rounding half away from zero; a
/// block whose `d` is zero (or not finite) stores zero codes, and a NaN
/// value codes as zero.
pub fn q8_0Row(input: []const f32, output: []u8) Error!void {
    if (input.len == 0 or input.len % q8_0_block != 0) return error.InvalidRowLength;
    if (output.len != input.len / q8_0_block * q8_0_block_bytes) return error.InvalidByteLength;
    var i: usize = 0;
    while (i < input.len / q8_0_block) : (i += 1) {
        const values = input[i * q8_0_block ..][0..q8_0_block];
        const block = output[i * q8_0_block_bytes ..][0..q8_0_block_bytes];
        var amax: f32 = 0;
        for (values) |x| if (!std.math.isNan(x)) {
            amax = @max(amax, @abs(x));
        };
        const d: f16 = @floatCast(amax / 127.0);
        std.mem.writeInt(u16, block[0..2], @bitCast(d), .little);
        const scale: f32 = d;
        for (values, block[2..]) |x, *code| {
            const q: f32 = if (scale == 0 or !std.math.isFinite(scale) or std.math.isNan(x)) 0 else std.math.clamp(@round(x / scale), -127, 127);
            code.* = @bitCast(@as(i8, @intFromFloat(q)));
        }
    }
}

const decode = @import("decode.zig");

test "Q8_0 round trip is within half a step per value" {
    var input: [96]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0x38);
    const random = prng.random();
    for (&input, 0..) |*x, j| x.* = (random.float(f32) - 0.5) * @as(f32, if (j < 32) 4 else if (j < 64) 0.002 else 300);
    var bytes: [3 * q8_0_block_bytes]u8 = undefined;
    try q8_0Row(&input, &bytes);
    var output: [96]f32 = undefined;
    try decode.row(8, &bytes, &output);
    for (0..3) |b| {
        const d: f32 = @as(f16, @bitCast(std.mem.readInt(u16, bytes[b * 34 ..][0..2], .little)));
        for (input[b * 32 ..][0..32], output[b * 32 ..][0..32]) |x, y| try std.testing.expect(@abs(x - y) <= d / 2 + @abs(x) * 1e-6);
    }
}

test "Q8_0 extremes, zero blocks, and refusals" {
    var input: [64]f32 = @splat(0);
    input[0] = -2.54;
    input[1] = 2.54;
    input[2] = 0.01;
    input[40] = std.math.nan(f32);
    var bytes: [68]u8 = undefined;
    try q8_0Row(&input, &bytes);
    try std.testing.expectEqual(@as(i8, -127), @as(i8, @bitCast(bytes[2])));
    try std.testing.expectEqual(@as(i8, 127), @as(i8, @bitCast(bytes[3])));
    try std.testing.expectEqual(@as(i8, 0), @as(i8, @bitCast(bytes[4]))); // just under half a step
    // The second block's NaN neither sets the scale nor a code.
    try std.testing.expectEqual(@as(u16, 0), std.mem.readInt(u16, bytes[34..36], .little));
    for (bytes[36..]) |code| try std.testing.expectEqual(@as(u8, 0), code);
    try std.testing.expectError(error.InvalidRowLength, q8_0Row(input[0..48], bytes[0..34]));
    try std.testing.expectError(error.InvalidByteLength, q8_0Row(input[0..32], bytes[0..33]));
}
