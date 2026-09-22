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
/// Kitty "disambiguate", bracketed paste, and focus in/out reports so a
/// finished turn can notify when the window is elsewhere.
const enter_modes = "\x1b[>1u\x1b[?2004h\x1b[?1004h";
const leave_modes = "\x1b[?1004l\x1b[?2004l\x1b[<u\x1b[0m\x1b[?25h";

pub const Terminal = struct {
    io: std.Io,
    saved: std.posix.termios,
    raw: std.posix.termios,
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
        var term: Terminal = .{ .io = io, .saved = saved, .raw = raw, .out = out };
        try term.acquire();
        return term;
    }
    pub fn deinit(self: *Terminal) void {
        // The trailing CRLF puts the shell prompt on a fresh line: the drawer
        // leaves the cursor at the end of the status bar, mid-row.
        self.out.writeAll(leave_modes ++ "\r\n") catch {};
        self.out.flush() catch {};
        std.posix.tcsetattr(0, .FLUSH, self.saved) catch {};
    }

    /// Raw mode and the modes above, on. Also how the lease is taken back
    /// after `release`.
    pub fn acquire(self: *Terminal) !void {
        try std.posix.tcsetattr(0, .FLUSH, self.raw);
        errdefer std.posix.tcsetattr(0, .FLUSH, self.saved) catch {};
        try self.out.writeAll(enter_modes);
        try self.out.flush();
    }

    /// Hands the terminal to a child (an external editor) as it was found:
    /// cooked mode, no protocol modes. `acquire` takes it back.
    pub fn release(self: *Terminal) !void {
        try self.out.writeAll(leave_modes);
        try self.out.flush();
        try std.posix.tcsetattr(0, .FLUSH, self.saved);
    }

    /// Puts `text` on the clipboard through OSC 52. Ghostty and most modern
    /// terminals honour it; the rest ignore it.
    pub fn copy(self: Terminal, alloc: std.mem.Allocator, text: []const u8) !void {
        const sequence = try osc52(alloc, text);
        defer alloc.free(sequence);
        try self.out.writeAll(sequence);
        try self.out.flush();
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

/// The most a Ctrl-X puts on the clipboard: terminals cap the OSC payload,
/// and an answer past this is not something to paste anyway.
pub const max_clipboard: usize = 256 * 1024;

/// The OSC 52 sequence that sets the clipboard (`c`) to `text`, base64 as
/// the protocol requires. Caller owns the result.
pub fn osc52(alloc: std.mem.Allocator, text: []const u8) ![]u8 {
    const encoder = std.base64.standard.Encoder;
    const out = try alloc.alloc(u8, "\x1b]52;c;".len + encoder.calcSize(text.len) + 1);
    @memcpy(out[0.."\x1b]52;c;".len], "\x1b]52;c;");
    _ = encoder.encode(out["\x1b]52;c;".len .. out.len - 1], text);
    out[out.len - 1] = 0x07;
    return out;
}

/// The most an external editor may hand back, the editor's own input limit.
pub const max_external_edit: usize = 128 * 1024;

/// Runs `command` (the `$VISUAL`/`$EDITOR` value, possibly with arguments)
/// on a temporary file holding `text`, and returns the file's content when
/// the editor exits, one trailing newline removed (editors add it). The
/// caller has released the terminal lease and takes it back afterwards; a
/// non-zero exit is an error and the text is left as it was. Caller owns
/// the result.
pub fn editExternally(alloc: std.mem.Allocator, io: std.Io, environ: *const std.process.Environ.Map, command: []const u8, text: []const u8) ![]u8 {
    var random: [8]u8 = undefined;
    io.randomSecure(&random) catch io.random(&random);
    const tmp = environ.get("TMPDIR") orelse "/tmp";
    const path = try std.fmt.allocPrintSentinel(alloc, "{s}/nuclis-edit-{x}.md", .{ std.mem.trimEnd(u8, tmp, "/"), &random }, 0);
    defer alloc.free(path);
    {
        const file = try std.Io.Dir.createFileAbsolute(io, path, .{});
        defer file.close(io);
        try file.writeStreamingAll(io, text);
    }
    defer std.Io.Dir.deleteFileAbsolute(io, path) catch {};
    // `$0` is the file: the value may carry its own arguments (`code -w`).
    const script = try std.fmt.allocPrint(alloc, "{s} \"$0\"", .{command});
    defer alloc.free(script);
    const argv = [_][]const u8{ "/bin/sh", "-c", script, path };
    var child = try std.process.spawn(io, .{
        .argv = &argv,
        .environ_map = environ,
        .stdin = .inherit,
        .stdout = .inherit,
        .stderr = .inherit,
        .expand_arg0 = .no_expand,
    });
    const term = try child.wait(io);
    switch (term) {
        .exited => |code| if (code != 0) return error.EditorFailed,
        else => return error.EditorFailed,
    }
    const edited = try std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(max_external_edit));
    if (edited.len > 0 and edited[edited.len - 1] == '\n') return alloc.realloc(edited, edited.len - 1);
    return edited;
}

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

test "the clipboard sequence is OSC 52 with the text in base64" {
    const alloc = std.testing.allocator;
    const hi = try osc52(alloc, "hi");
    defer alloc.free(hi);
    try std.testing.expectEqualStrings("\x1b]52;c;aGk=\x07", hi);
    const empty = try osc52(alloc, "");
    defer alloc.free(empty);
    try std.testing.expectEqualStrings("\x1b]52;c;\x07", empty);
    // A newline and a non-ASCII byte are payload, never a break in the sequence.
    const multi = try osc52(alloc, "a\n中");
    defer alloc.free(multi);
    try std.testing.expect(std.mem.indexOfScalar(u8, multi[0 .. multi.len - 1], '\n') == null);
    try std.testing.expect(std.mem.endsWith(u8, multi, "\x07"));
}

test "an external editor round trip: `true` returns the text, a writer replaces it, a failure keeps it" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    var environ = std.process.Environ.Map.init(alloc);
    defer environ.deinit();
    try environ.put("PATH", "/usr/bin:/bin");
    const same = try editExternally(alloc, io, &environ, "true", "keep\nme");
    defer alloc.free(same);
    try std.testing.expectEqualStrings("keep\nme", same);
    // The editor's argument is the file, whatever else the value carries.
    const replaced = try editExternally(alloc, io, &environ, "sh -c 'printf \"new text\\n\" > \"$0\"'", "old");
    defer alloc.free(replaced);
    try std.testing.expectEqualStrings("new text", replaced);
    try std.testing.expectError(error.EditorFailed, editExternally(alloc, io, &environ, "false", "old"));
}
