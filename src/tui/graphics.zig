//! Inline image previews through the kitty graphics protocol: a decoded RGB
//! image, downscaled on the host, becomes one direct transmission
//! (`f=24`, base64 in 4096-byte chunks) placed over a box of cells with the
//! cursor left where it was (`C=1`). The transcript reserves the box's rows
//! and carries the sequence as one raw row; nothing here touches the
//! terminal. Whether a terminal draws it is decided by the environment, as
//! the screen's other capabilities are; under tmux the sequence is wrapped
//! in the DCS passthrough. Never part of the golden tests.
const std = @import("std");
const inference = @import("inference");

const Rgb8 = inference.vision.image.Rgb8;
const Allocator = std.mem.Allocator;

/// The longest side of the transmitted copy, in pixels.
pub const max_side: u32 = 512;
/// The tallest box a preview may take, in rows.
pub const max_rows: usize = 12;
/// A cell is assumed twice as tall as wide, the common terminal aspect.
pub const cell_aspect: f64 = 2.0;
const chunk: usize = 4096;

/// Whether the terminal is one that draws kitty graphics (Ghostty, kitty),
/// by environment; `NUCLIS_NO_PREVIEW=1` turns it off.
pub fn enabled(environ: *const std.process.Environ.Map) bool {
    if (environ.get("NUCLIS_NO_PREVIEW")) |v| if (!std.mem.eql(u8, v, "0")) return false;
    if (environ.get("GHOSTTY_RESOURCES_DIR") != null or environ.get("KITTY_WINDOW_ID") != null) return true;
    if (environ.get("TERM_PROGRAM")) |p| if (std.ascii.eqlIgnoreCase(p, "ghostty")) return true;
    const term = environ.get("TERM") orelse "";
    return std.mem.indexOf(u8, term, "ghostty") != null or std.mem.indexOf(u8, term, "kitty") != null;
}

/// The cell box a `width × height` image takes: at most `max_rows` tall and
/// `max_columns` wide, aspect kept under `cell_aspect`.
pub const Box = struct { rows: usize, columns: usize };

pub fn box(width: u32, height: u32, max_columns: usize) Box {
    if (width == 0 or height == 0) return .{ .rows = 1, .columns = 1 };
    const w: f64 = @floatFromInt(width);
    const h: f64 = @floatFromInt(height);
    // Rows from the height at ~20 px a row, capped; columns from the aspect.
    var rows: f64 = @min(@as(f64, @floatFromInt(max_rows)), @max(1.0, @ceil(h / 20.0)));
    var columns: f64 = @max(1.0, @round(rows * cell_aspect * w / h));
    const cap: f64 = @floatFromInt(@max(max_columns, 1));
    if (columns > cap) {
        columns = cap;
        rows = @max(1.0, @round(columns * h / (w * cell_aspect)));
    }
    return .{ .rows = @intFromFloat(rows), .columns = @intFromFloat(columns) };
}

/// A box-filtered copy whose longest side is at most `max_side`; the source
/// itself (copied) when it already fits.
pub fn scaleDown(alloc: Allocator, source: Rgb8) !Rgb8 {
    const longest = @max(source.width, source.height);
    if (longest <= max_side) return .{ .width = source.width, .height = source.height, .pixels = try alloc.dupe(u8, source.pixels) };
    const width: u32 = @max(1, @as(u32, @intCast((@as(u64, source.width) * max_side) / longest)));
    const height: u32 = @max(1, @as(u32, @intCast((@as(u64, source.height) * max_side) / longest)));
    const pixels = try alloc.alloc(u8, @as(usize, width) * height * 3);
    errdefer alloc.free(pixels);
    var y: usize = 0;
    while (y < height) : (y += 1) {
        const y0 = (y * source.height) / height;
        const y1 = @max(y0 + 1, ((y + 1) * source.height) / height);
        var x: usize = 0;
        while (x < width) : (x += 1) {
            const x0 = (x * source.width) / width;
            const x1 = @max(x0 + 1, ((x + 1) * source.width) / width);
            var sum: [3]u64 = .{ 0, 0, 0 };
            var sy = y0;
            while (sy < y1) : (sy += 1) {
                var sx = x0;
                while (sx < x1) : (sx += 1) {
                    const p = source.at(sx, sy);
                    inline for (0..3) |c| sum[c] += p[c];
                }
            }
            const n = (y1 - y0) * (x1 - x0);
            const at = (y * width + x) * 3;
            inline for (0..3) |c| pixels[at + c] = @intCast(sum[c] / n);
        }
    }
    return .{ .width = width, .height = height, .pixels = pixels };
}

/// The transmission-and-placement sequence for `image` over `b`: the pixels
/// as base64 in `chunk`-sized commands (`m=1` until the last), quiet (`q=2`),
/// cursor kept (`C=1`). `tmux` wraps every command in the passthrough DCS.
pub fn sequence(alloc: Allocator, image: Rgb8, b: Box, tmux: bool) ![]u8 {
    const encoder = std.base64.standard.Encoder;
    const encoded = try alloc.alloc(u8, encoder.calcSize(image.pixels.len));
    defer alloc.free(encoded);
    _ = encoder.encode(encoded, image.pixels);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    var at: usize = 0;
    var first = true;
    while (first or at < encoded.len) {
        const end = @min(at + chunk, encoded.len);
        const more: u8 = if (end < encoded.len) '1' else '0';
        var command: std.Io.Writer.Allocating = .init(alloc);
        defer command.deinit();
        try command.writer.writeAll("\x1b_G");
        if (first) try command.writer.print("a=T,f=24,s={d},v={d},c={d},r={d},C=1,q=2,", .{ image.width, image.height, b.columns, b.rows });
        try command.writer.print("m={c};", .{more});
        try command.writer.writeAll(encoded[at..end]);
        try command.writer.writeAll("\x1b\\");
        if (tmux) try passthrough(alloc, &out, command.written()) else try out.appendSlice(alloc, command.written());
        at = end;
        first = false;
    }
    return out.toOwnedSlice(alloc);
}

/// tmux's DCS passthrough: `ESC P tmux ; <body with ESC doubled> ESC \`.
fn passthrough(alloc: Allocator, out: *std.ArrayList(u8), body: []const u8) !void {
    try out.appendSlice(alloc, "\x1bPtmux;");
    for (body) |c| {
        if (c == 0x1b) try out.append(alloc, 0x1b);
        try out.append(alloc, c);
    }
    try out.appendSlice(alloc, "\x1b\\");
}

// ----- tests -----

const testing = std.testing;

test "the box keeps the aspect under the row cap and the width cap" {
    const tall = box(400, 800, 120);
    try testing.expectEqual(max_rows, tall.rows);
    try testing.expectEqual(@as(usize, 12), tall.columns);
    const wide = box(2000, 200, 40);
    try testing.expectEqual(@as(usize, 40), wide.columns);
    try testing.expectEqual(@as(usize, 2), wide.rows);
    const tiny = box(16, 16, 80);
    try testing.expectEqual(@as(usize, 1), tiny.rows);
    try testing.expectEqual(@as(usize, 2), tiny.columns);
}

test "a large image is box-filtered to the longest side, a small one copied" {
    const a = testing.allocator;
    const pixels = try a.alloc(u8, 1024 * 2 * 3);
    defer a.free(pixels);
    for (pixels, 0..) |*p, i| p.* = if ((i / 3) % 2 == 0) 0 else 200; // alternating columns
    const big: Rgb8 = .{ .width = 1024, .height = 2, .pixels = pixels };
    var small = try scaleDown(a, big);
    defer small.deinit(a);
    try testing.expectEqual(@as(u32, 512), small.width);
    try testing.expectEqual(@as(u32, 1), small.height);
    // Each output pixel averages a 2×2 block of 0 and 200.
    try testing.expectEqual([3]u8{ 100, 100, 100 }, small.at(0, 0));
    var same = try scaleDown(a, small);
    defer same.deinit(a);
    try testing.expectEqualSlices(u8, small.pixels, same.pixels);
}

test "the sequence is chunked, quiet, cursor-keeping, and tmux-wrapped on request" {
    const a = testing.allocator;
    const pixels = try a.alloc(u8, 2000 * 3);
    defer a.free(pixels);
    @memset(pixels, 7);
    const image: Rgb8 = .{ .width = 2000, .height = 1, .pixels = pixels };
    const plain = try sequence(a, image, .{ .rows = 3, .columns = 40 }, false);
    defer a.free(plain);
    try testing.expect(std.mem.startsWith(u8, plain, "\x1b_Ga=T,f=24,s=2000,v=1,c=40,r=3,C=1,q=2,m=1;"));
    // 6000 bytes → 8000 base64 → two chunks, the last with m=0.
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, plain, "\x1b_G"));
    try testing.expect(std.mem.indexOf(u8, plain, "\x1b_Gm=0;") != null);
    const wrapped = try sequence(a, image, .{ .rows = 3, .columns = 40 }, true);
    defer a.free(wrapped);
    try testing.expect(std.mem.startsWith(u8, wrapped, "\x1bPtmux;\x1b\x1b_G"));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, wrapped, "\x1bPtmux;"));
}

test "the preview is decided by the environment" {
    var environ = std.process.Environ.Map.init(testing.allocator);
    defer environ.deinit();
    try testing.expect(!enabled(&environ));
    try environ.put("TERM_PROGRAM", "ghostty");
    try testing.expect(enabled(&environ));
    try environ.put("NUCLIS_NO_PREVIEW", "1");
    try testing.expect(!enabled(&environ));
}
