//! Pure, bounded Xet chunk decoding. No networking, allocations, or platform I/O.
//! The caller owns output and scratch (128 KiB each); returned slices borrow output.
//! LZ4 uses frames, including checksums, rather than a bare block payload.
const std = @import("std");
pub const max_chunk = 128 * 1024;
pub const Error = error{ InvalidChunk, UnsupportedCompression, InvalidLz4, UnsupportedLz4, ChecksumMismatch };

pub const Cursor = struct {
    bytes: []const u8,
    pub fn take(c: *Cursor, n: usize) Error![]const u8 {
        if (n > c.bytes.len) return error.InvalidChunk;
        const result = c.bytes[0..n];
        c.bytes = c.bytes[n..];
        return result;
    }
    fn int(c: *Cursor, comptime T: type) Error!T {
        const b = try c.take(@divExact(@bitSizeOf(T), 8));
        return std.mem.readInt(T, b[0..@divExact(@bitSizeOf(T), 8)], .little);
    }
};

/// Advances past one chunk without decompressing it: the header carries the
/// packed size, so chunks outside a term cost a bounds check, not a decode.
pub fn skip(c: *Cursor) Error!void {
    const h = try c.take(8);
    const packed_size = std.mem.readInt(u24, h[1..4], .little);
    if (h[0] != 0 or packed_size > max_chunk + 1024) return error.InvalidChunk;
    _ = try c.take(packed_size);
}

pub fn decode(c: *Cursor, output: []u8, scratch: []u8) Error![]const u8 {
    const h = try c.take(8);
    const packed_size = std.mem.readInt(u24, h[1..4], .little);
    const size = std.mem.readInt(u24, h[5..8], .little);
    if (h[0] != 0 or size == 0 or size > max_chunk or size > output.len or size > scratch.len or packed_size > max_chunk + 1024)
        return error.InvalidChunk;
    const data = try c.take(packed_size);
    switch (h[4]) {
        0 => {
            if (data.len != size) return error.InvalidChunk;
            @memcpy(output[0..size], data);
        },
        1 => try frame(data, output[0..size]),
        2 => {
            try frame(data, scratch[0..size]);
            // Remainder bytes belong to the first lanes, not a trailing tail.
            var offset: usize = 0;
            for (0..4) |lane| {
                const n = size / 4 + @intFromBool(lane < size % 4);
                for (0..n) |row| output[row * 4 + lane] = scratch[offset + row];
                offset += n;
            }
        },
        else => return error.UnsupportedCompression,
    }
    return output[0..size];
}

fn length(c: *Cursor, nibble: u8, bound: usize) Error!usize {
    var n: usize = nibble;
    if (n == 15) while (true) {
        const extra = try c.int(u8);
        if (n > bound or extra > bound - n) return error.InvalidLz4;
        n += extra;
        if (extra != 255) break;
    };
    return n;
}

fn block(data: []const u8, out: []u8, start: usize, independent: bool) Error!usize {
    var c: Cursor = .{ .bytes = data };
    var pos = start;
    while (c.bytes.len > 0) {
        const token = try c.int(u8);
        const literals = try length(&c, token >> 4, out.len - pos);
        if (literals > out.len - pos) return error.InvalidLz4;
        @memcpy(out[pos..][0..literals], try c.take(literals));
        pos += literals;
        if (c.bytes.len == 0) return pos;
        const distance = try c.int(u16);
        if (distance == 0 or distance > pos - (if (independent) start else 0)) return error.InvalidLz4;
        const match_len = (try length(&c, token & 15, out.len - pos)) + 4;
        if (match_len > out.len - pos) return error.InvalidLz4;
        // Forward copying is essential: matches can overlap their own output.
        for (0..match_len) |_| {
            out[pos] = out[pos - distance];
            pos += 1;
        }
    }
    return error.InvalidLz4;
}

fn frame(data: []const u8, out: []u8) Error!void {
    var c: Cursor = .{ .bytes = data };
    if (try c.int(u32) != 0x184d2204) return error.UnsupportedLz4;
    const flags = try c.int(u8);
    const descriptor = try c.int(u8);
    if (flags >> 6 != 1 or flags & 2 != 0 or descriptor & 0x8f != 0) return error.InvalidLz4;
    if (flags & 1 != 0) return error.UnsupportedLz4; // External dictionaries aren't Xet chunk data.
    const block_code = (descriptor >> 4) & 7;
    if (block_code < 4) return error.InvalidLz4;
    const max_block: usize = @as(usize, 1) << @as(u6, @intCast(8 + 2 * block_code));
    if (flags & 8 != 0 and try c.int(u64) != out.len) return error.InvalidLz4;
    const descriptor_end = data.len - c.bytes.len;
    if (try c.int(u8) != @as(u8, @truncate(std.hash.XxHash32.hash(0, data[4..descriptor_end]) >> 8))) return error.ChecksumMismatch;
    var pos: usize = 0;
    while (true) {
        const encoded = try c.int(u32);
        if (encoded == 0) break;
        const n = encoded & 0x7fffffff;
        if (n > max_block) return error.InvalidLz4;
        const payload = try c.take(n);
        if (flags & 16 != 0 and try c.int(u32) != std.hash.XxHash32.hash(0, payload)) return error.ChecksumMismatch;
        const start = pos;
        if (encoded & 0x80000000 != 0) {
            if (n > out.len - pos) return error.InvalidLz4;
            @memcpy(out[pos..][0..n], payload);
            pos += n;
        } else pos = try block(payload, out, pos, flags & 32 != 0);
        if (pos - start > max_block) return error.InvalidLz4;
    }
    if (pos != out.len) return error.InvalidLz4;
    if (flags & 4 != 0 and try c.int(u32) != std.hash.XxHash32.hash(0, out)) return error.ChecksumMismatch;
    if (c.bytes.len != 0) return error.InvalidLz4;
}

test "raw chunks validate sizes, versions, and compression" {
    var out: [32]u8 = undefined;
    var scratch: [32]u8 = undefined;
    var c: Cursor = .{ .bytes = &.{ 0, 4, 0, 0, 0, 4, 0, 0, 'G', 'G', 'U', 'F' } };
    try std.testing.expectEqualStrings("GGUF", try decode(&c, &out, &scratch));
    try std.testing.expectEqual(@as(usize, 0), c.bytes.len);
    c.bytes = &.{ 0, 3, 0, 0, 0, 4, 0, 0, 1, 2, 3 };
    try std.testing.expectError(error.InvalidChunk, decode(&c, &out, &scratch));
    c.bytes = &.{ 0, 0, 0, 0, 99, 4, 0, 0 };
    try std.testing.expectError(error.UnsupportedCompression, decode(&c, &out, &scratch));
    c.bytes = &.{ 0, 4, 0, 0, 0, 4, 0, 0, 'G', 'G', 'U', 'F', 0, 1, 0, 0, 0, 1, 0, 0, 'x' };
    try skip(&c);
    try std.testing.expectEqualStrings("x", try decode(&c, &out, &scratch));
    try std.testing.expectError(error.InvalidChunk, skip(&c));
}

test "LZ4 matches overlap and reject invalid distances" {
    var out: [16]u8 = undefined;
    const n = try block(&.{ 0x13, 'a', 1, 0, 0x50, 'b', 'c', 'd', 'e', 'f' }, &out, 0, true);
    try std.testing.expectEqualStrings("aaaaaaaabcdef", out[0..n]);
    try std.testing.expectError(error.InvalidLz4, block(&.{ 0, 0, 0 }, &out, 0, true));
    try std.testing.expectError(error.InvalidChunk, block(&.{0xf0}, &out, 0, true));
}

test "framed LZ4 and byte grouping with remainder and checksums" {
    // Hand-constructed frame: one stored block, header and content checksums.
    // Expected bytes are independent of any encoder implementation.
    var data: [8 + 7 + 4 + 7 + 4 + 4]u8 = @splat(0);
    data[1] = data.len - 8;
    data[4] = 2;
    data[5] = 7;
    std.mem.writeInt(u32, data[8..12], 0x184d2204, .little);
    data[12] = 0x64;
    data[13] = 0x40;
    data[14] = @truncate(std.hash.XxHash32.hash(0, data[12..14]) >> 8);
    std.mem.writeInt(u32, data[15..19], 0x80000007, .little);
    @memcpy(data[19..26], "aebfcgd");
    std.mem.writeInt(u32, data[30..34], std.hash.XxHash32.hash(0, data[19..26]), .little);
    var output: [7]u8 = undefined;
    var scratch: [7]u8 = undefined;
    var c: Cursor = .{ .bytes = &data };
    try std.testing.expectEqualStrings("abcdefg", try decode(&c, &output, &scratch));
    for (0..data.len) |end| {
        c.bytes = data[0..end];
        if (decode(&c, &output, &scratch)) |_| return error.TestUnexpectedResult else |_| {}
    }
    data[33] ^= 1;
    c.bytes = &data;
    try std.testing.expectError(error.ChecksumMismatch, decode(&c, &output, &scratch));
}
