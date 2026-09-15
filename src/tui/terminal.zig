//! POSIX terminal lease. Restoration runs before stdout's buffered writer dies.
//! Input uses injected Io; termios, poll, and window size are OS controls only.
//!
//! The lease does NOT use the alternate screen: the conversation lives in the
//! normal buffer, so the terminal's own scrolling, mouse wheel, and scrollback
//! search work without the chat reimplementing them. The chat only repaints a
//! small live region above the cursor (see root.zig); completed turns are
//! written once above it and left to scroll away. On entry the lease requests
//! the kitty keyboard protocol's "disambiguate" level (`CSI > 1 u`) so
//! Shift+Enter is reportable; terminals that do not support it ignore the
//! request. It also enables bracketed paste (`CSI ? 2004 h`), so a paste
//! arrives between `CSI 200~` and `CSI 201~` markers and its newlines are
//! text rather than Enter presses. Cursor visibility is left to the drawer,
//! which hides it while painting and shows it at the editor position.
const std = @import("std");
const builtin = @import("builtin");
pub const Size = struct { rows: usize = 24, columns: usize = 80 };
pub const Terminal = struct {
    io: std.Io,
    saved: std.posix.termios,
    out: *std.Io.Writer,
    pub fn init(io: std.Io, out: *std.Io.Writer) !Terminal {
        if (!try std.Io.File.stdin().isTty(io) or !try std.Io.File.stdout().isTty(io)) return error.ChatRequiresTerminal;
        const saved = try std.posix.tcgetattr(0);
        var raw = saved;
        raw.lflag.ECHO = false;
        raw.lflag.ICANON = false;
        raw.lflag.ISIG = false;
        raw.lflag.IEXTEN = false;
        raw.iflag.IXON = false;
        raw.iflag.ICRNL = false;
        raw.cc[@intFromEnum(std.posix.V.MIN)] = 1;
        raw.cc[@intFromEnum(std.posix.V.TIME)] = 0;
        try std.posix.tcsetattr(0, .FLUSH, raw);
        errdefer std.posix.tcsetattr(0, .FLUSH, saved) catch {};
        // Kitty "disambiguate", bracketed paste, and focus in/out reports so a
        // finished turn can notify when the window is elsewhere.
        try out.writeAll("\x1b[>1u\x1b[?2004h\x1b[?1004h");
        try out.flush();
        return .{ .io = io, .saved = saved, .out = out };
    }
    pub fn deinit(self: *Terminal) void {
        // The trailing CRLF puts the shell prompt on a fresh line: the drawer
        // leaves the cursor at the end of the status bar, mid-row.
        self.out.writeAll("\x1b[?1004l\x1b[?2004l\x1b[<u\x1b[0m\x1b[?25h\r\n") catch {};
        self.out.flush() catch {};
        std.posix.tcsetattr(0, .FLUSH, self.saved) catch {};
    }
    pub fn size(_: Terminal) Size {
        var value: std.posix.winsize = .{ .row = 0, .col = 0, .xpixel = 0, .ypixel = 0 };
        // TIOCGWINSZ from the platform terminal ABI; no model/backend dependency.
        const request: c_ulong = switch (builtin.os.tag) {
            .macos => 0x40087468,
            .linux => 0x5413,
            else => return .{},
        };
        if (std.posix.system.ioctl(1, request, &value) != 0) return .{};
        return .{ .rows = std.math.clamp(value.row, 8, 200), .columns = std.math.clamp(value.col, 20, 400) };
    }
    pub fn read(self: Terminal, bytes: []u8, timeout: i32) !usize {
        var fds = [_]std.posix.pollfd{.{ .fd = 0, .events = std.posix.POLL.IN, .revents = 0 }};
        if (try std.posix.poll(&fds, timeout) == 0) return 0;
        const n = try std.Io.File.stdin().readStreaming(self.io, &.{bytes});
        if (n == 0) return error.EndOfStream;
        return n;
    }

    /// One desktop notification (OSC 9). `message` is a host constant; the
    /// terminals that support OSC 9 show it, the rest ignore it.
    pub fn notify(self: Terminal, message: []const u8) !void {
        try self.out.writeAll("\x1b]9;");
        try self.out.writeAll(message);
        try self.out.writeAll("\x07");
        try self.out.flush();
    }
};

/// Whether to send desktop notifications: off for a dumb terminal or when
/// `NUCLIS_NO_NOTIFY` is set to anything but `0`.
pub fn notificationsEnabled(environ: *const std.process.Environ.Map) bool {
    const term = environ.get("TERM") orelse "";
    if (std.mem.eql(u8, term, "dumb")) return false;
    const value = environ.get("NUCLIS_NO_NOTIFY") orelse return true;
    return std.mem.eql(u8, value, "0");
}

/// A finished turn notifies only when the window was elsewhere and the user
/// did not cancel it. Pure, so the decision is tested without a terminal.
pub fn shouldNotify(focused: bool, enabled: bool, completed: bool) bool {
    return enabled and !focused and completed;
}

test "focus and notification decisions" {
    try std.testing.expect(shouldNotify(false, true, true));
    try std.testing.expect(!shouldNotify(true, true, true)); // focused
    try std.testing.expect(!shouldNotify(false, false, true)); // disabled
    try std.testing.expect(!shouldNotify(false, true, false)); // cancelled

    var map = std.process.Environ.Map.init(std.testing.allocator);
    defer map.deinit();
    try std.testing.expect(notificationsEnabled(&map));
    try map.put("TERM", "dumb");
    try std.testing.expect(!notificationsEnabled(&map));
    try map.put("TERM", "xterm-256color");
    try map.put("NUCLIS_NO_NOTIFY", "1");
    try std.testing.expect(!notificationsEnabled(&map));
    try map.put("NUCLIS_NO_NOTIFY", "0");
    try std.testing.expect(notificationsEnabled(&map));
}
