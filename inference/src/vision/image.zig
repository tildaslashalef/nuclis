//! Decoded images for the vision projectors: tightly packed 8-bit RGB rows,
//! top row first. Two decoders: a P6 PPM parser (fixtures, tests, any
//! platform) and, when the macOS bridges are built, ImageIO for the
//! platform's formats. Bounds are host constants, never model-supplied.
const std = @import("std");
const build_options = @import("build_options");

/// The bridge is compiled with the Metal one: both are the macOS bridges.
pub const bridge_enabled = build_options.metal;

/// Encoded bytes accepted from a file.
pub const max_bytes: usize = 32 << 20;
/// Decoded pixels accepted (width × height).
pub const max_pixels: usize = 64 << 20;

pub const Error = error{
    ImageTooLarge,
    MalformedImage,
    UnsupportedImageFormat,
    OutOfMemory,
};

pub const Rgb8 = struct {
    width: u32,
    height: u32,
    /// `width * height * 3` bytes, row-major, owned.
    pixels: []u8,

    pub fn deinit(self: *Rgb8, alloc: std.mem.Allocator) void {
        alloc.free(self.pixels);
        self.* = undefined;
    }

    pub fn at(self: Rgb8, x: usize, y: usize) [3]u8 {
        const i = (y * self.width + x) * 3;
        return self.pixels[i..][0..3].*;
    }
};

/// Decodes `bytes` by sniffing the format: P6 first, then the bridge.
pub fn decode(alloc: std.mem.Allocator, bytes: []const u8) Error!Rgb8 {
    if (bytes.len > max_bytes) return error.ImageTooLarge;
    if (bytes.len >= 2 and bytes[0] == 'P' and bytes[1] == '6') return decodePpm(alloc, bytes);
    if (!bridge_enabled) return error.UnsupportedImageFormat;
    var width: u32 = 0;
    var height: u32 = 0;
    var pixels: [*]u8 = undefined;
    switch (nu_image_decode(bytes.ptr, bytes.len, @intCast(max_pixels), &width, &height, &pixels)) {
        0 => {},
        1 => return error.UnsupportedImageFormat,
        2 => return error.ImageTooLarge,
        else => return error.OutOfMemory,
    }
    defer nu_image_free(pixels);
    const count = @as(usize, width) * height * 3;
    const owned = try alloc.dupe(u8, pixels[0..count]);
    return .{ .width = width, .height = height, .pixels = owned };
}

extern fn nu_image_decode([*]const u8, usize, u32, *u32, *u32, *[*]u8) c_int;
extern fn nu_image_free([*]u8) void;

/// The binary PPM form: `P6`, whitespace-separated width, height, and a
/// maxval of 255 (comments allowed between fields), one whitespace byte,
/// then the raw RGB rows.
pub fn decodePpm(alloc: std.mem.Allocator, bytes: []const u8) Error!Rgb8 {
    if (bytes.len > max_bytes) return error.ImageTooLarge;
    if (bytes.len < 2 or bytes[0] != 'P' or bytes[1] != '6') return error.MalformedImage;
    var cursor: usize = 2;
    const width = try ppmField(bytes, &cursor);
    const height = try ppmField(bytes, &cursor);
    const maxval = try ppmField(bytes, &cursor);
    if (maxval != 255 or width == 0 or height == 0) return error.UnsupportedImageFormat;
    if (cursor >= bytes.len or !std.ascii.isWhitespace(bytes[cursor])) return error.MalformedImage;
    cursor += 1;
    if (@as(u64, width) * height > max_pixels) return error.ImageTooLarge;
    const count = @as(usize, width) * height * 3;
    if (bytes.len - cursor < count) return error.MalformedImage;
    const pixels = try alloc.dupe(u8, bytes[cursor..][0..count]);
    return .{ .width = width, .height = height, .pixels = pixels };
}

fn ppmField(bytes: []const u8, cursor: *usize) Error!u32 {
    var i = cursor.*;
    while (i < bytes.len) {
        if (bytes[i] == '#') {
            while (i < bytes.len and bytes[i] != '\n') i += 1;
        } else if (std.ascii.isWhitespace(bytes[i])) {
            i += 1;
        } else break;
    }
    const start = i;
    while (i < bytes.len and std.ascii.isDigit(bytes[i])) i += 1;
    if (i == start or i - start > 8) return error.MalformedImage;
    cursor.* = i;
    return std.fmt.parseInt(u32, bytes[start..i], 10) catch return error.MalformedImage;
}

/// Writes `image` as P6 (tests and fixtures).
pub fn encodePpm(alloc: std.mem.Allocator, image: Rgb8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    try out.print(alloc, "P6\n{d} {d}\n255\n", .{ image.width, image.height });
    try out.appendSlice(alloc, image.pixels);
    return out.toOwnedSlice(alloc);
}

test "P6 parses fields, comments, and the single separator byte" {
    const bytes = "P6\n# a comment\n2 1\n255\n\x01\x02\x03\x04\x05\x06";
    var image = try decodePpm(std.testing.allocator, bytes);
    defer image.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 2), image.width);
    try std.testing.expectEqual(@as(u32, 1), image.height);
    try std.testing.expectEqual([3]u8{ 4, 5, 6 }, image.at(1, 0));
    const round = try encodePpm(std.testing.allocator, image);
    defer std.testing.allocator.free(round);
    var again = try decode(std.testing.allocator, round);
    defer again.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(u8, image.pixels, again.pixels);
}

test "P6 rejects truncated data, other maxvals, absurd sizes, and other magics" {
    try std.testing.expectError(error.MalformedImage, decodePpm(std.testing.allocator, "P6\n2 1\n255\n\x01\x02\x03"));
    try std.testing.expectError(error.UnsupportedImageFormat, decodePpm(std.testing.allocator, "P6\n1 1\n65535\n\x00\x00\x00\x00\x00\x00"));
    try std.testing.expectError(error.ImageTooLarge, decodePpm(std.testing.allocator, "P6\n99999 99999\n255\n"));
    try std.testing.expectError(error.MalformedImage, decodePpm(std.testing.allocator, "P5\n1 1\n255\n\x00"));
    try std.testing.expectError(error.MalformedImage, decodePpm(std.testing.allocator, "P6\n0x1 1\n255\n"));
    if (!bridge_enabled) try std.testing.expectError(error.UnsupportedImageFormat, decode(std.testing.allocator, "\x89PNG\r\n"));
}

test "the bridge decodes a PNG and rejects junk" {
    if (!bridge_enabled) return error.SkipZigTest;
    // A 2×1 RGB PNG: red then green.
    const png = [_]u8{
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
        0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x7B, 0x40, 0xE8,
        0xDD, 0x00, 0x00, 0x00, 0x0F, 0x49, 0x44, 0x41, 0x54, 0x78, 0xDA, 0x63, 0xF8, 0xCF, 0xC0, 0xC0,
        0xF0, 0x9F, 0x01, 0x00, 0x07, 0xFF, 0x01, 0xFF, 0xB8, 0x04, 0x35, 0xE0, 0x00, 0x00, 0x00, 0x00,
        0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    };
    var image = try decode(std.testing.allocator, &png);
    defer image.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 2), image.width);
    try std.testing.expectEqual(@as(u32, 1), image.height);
    try std.testing.expectEqual([3]u8{ 255, 0, 0 }, image.at(0, 0));
    try std.testing.expectEqual([3]u8{ 0, 255, 0 }, image.at(1, 0));
    try std.testing.expectError(error.UnsupportedImageFormat, decode(std.testing.allocator, "not an image at all"));
}
