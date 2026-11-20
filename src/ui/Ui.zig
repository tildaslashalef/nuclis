//! Rendering settings for one stream of one invocation: whether and how
//! deep to colour, which glyph set, and the terminal's width (null off a
//! terminal: no limit, nothing truncated).
const std = @import("std");
const theme = @import("../tui/theme.zig");
const tui_style = @import("../tui/style.zig");
const style = @import("style.zig");

const Ui = @This();

/// The theme and whether it is painted at all.
paint: tui_style.Style = .none,
glyphs: *const style.Glyphs = &style.unicode,
width: ?usize = null,

/// No colour, Unicode glyphs, no width limit: the bytes the tests pin.
pub const plain: Ui = .{};

/// Settings for output written to `file`: colour by `Style.detect` (a
/// terminal, `NO_COLOR`, `TERM`), the glyph set by the locale or
/// `NUCLIS_ASCII`, the width from `COLUMNS` or the terminal.
pub fn detect(environ: *const std.process.Environ.Map, io: std.Io, file: std.Io.File) Ui {
    const paint = tui_style.Style.detect(environ, io, file);
    const tty = file.isTty(io) catch false;
    return .{
        .paint = paint,
        .glyphs = if (paint.theme.glyph_set == .ascii) &style.ascii else &style.unicode,
        .width = if (tty) terminalWidth(environ, io, file) else null,
    };
}

/// For a `--json` run: nothing painted, nothing cut.
pub fn forJson(self: Ui) Ui {
    return .{ .glyphs = self.glyphs };
}

/// The escape that starts a span of `role`, or nothing.
pub fn on(self: Ui, role: style.Role) []const u8 {
    const kind = role.kind() orelse return "";
    return self.paint.on(kind);
}

/// The escape that ends a span of `role`, or nothing.
pub fn off(self: Ui, role: style.Role) []const u8 {
    if (role == .text) return "";
    return self.paint.off();
}

/// `COLUMNS` wins (it pins a width for tests and scripts); else the
/// terminal's own, else none.
fn terminalWidth(environ: *const std.process.Environ.Map, io: std.Io, file: std.Io.File) ?usize {
    if (environ.get("COLUMNS")) |c| {
        if (std.fmt.parseInt(usize, c, 10)) |n| {
            if (n >= 20) return n;
        } else |_| {}
    }
    _ = io;
    var ws: std.posix.winsize = .{ .row = 0, .col = 0, .xpixel = 0, .ypixel = 0 };
    // TIOCGWINSZ, as `tui/terminal.zig` asks it, on this stream's descriptor.
    const request: c_ulong = switch (@import("builtin").target.os.tag) {
        .macos => 0x40087468,
        .linux => 0x5413,
        else => return null,
    };
    if (std.posix.system.ioctl(file.handle, request, &ws) != 0 or ws.col == 0) return null;
    return ws.col;
}

test "a stream that is not a terminal gets no colour and no width" {
    const io = std.testing.io;
    var env: std.process.Environ.Map = .init(std.testing.allocator);
    defer env.deinit();
    try env.put("COLORTERM", "truecolor");
    try env.put("TERM", "xterm-256color");
    try env.put("COLUMNS", "120");
    const devnull = try std.Io.Dir.cwd().openFile(io, "/dev/null", .{ .mode = .write_only });
    defer devnull.close(io);
    const ui = detect(&env, io, devnull);
    try std.testing.expectEqualStrings("", ui.on(.ok));
    try std.testing.expectEqual(@as(?usize, null), ui.width);
    const lit: Ui = .{ .paint = .{ .theme = .{ .kind = .truecolor }, .enabled = true } };
    try std.testing.expectEqualStrings("\x1b[38;2;184;187;38m", lit.on(.ok));
    try std.testing.expectEqualStrings("\x1b[0m", lit.off(.ok));
    try std.testing.expectEqualStrings("", lit.on(.text));
    try std.testing.expectEqualStrings("", lit.off(.text));
}
