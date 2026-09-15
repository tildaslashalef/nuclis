//! Terminal styling for command output: the agent's palette (`theme.zig`,
//! gruvbox dark by default) applied to the text reports of every command.
//!
//! Colors are on only when the output is a terminal and the environment
//! allows them (`Theme.detect`: `NO_COLOR` or an unknown `TERM` disables
//! them entirely here, since a report with bare bold attributes helps no
//! one); `--json` output and piped output are never styled, so scripts and
//! the tests see exactly the same bytes. Every span ends with `reset`, and
//! callers pad the text before wrapping it in escapes (`"{s}{s: <12}{s}"`)
//! so column alignment never depends on the escape bytes.
//!
//! Zig notes. `Style` is a two-field value passed explicitly to every
//! renderer, like the allocator and `Io`: no global decides whether output
//! is colored, and a test constructs `.none` to pin the plain form.
const std = @import("std");
const theme = @import("theme.zig");

pub const Kind = theme.Style;
/// The configured palette name (`agent.theme`); command reports take the
/// default, since they are painted before any configuration is read.
pub const Name = theme.Name;

pub const Style = struct {
    theme: theme.Theme = .{ .kind = .plain },
    enabled: bool = false,

    /// No styling at all: the bytes the tests pin.
    pub const none: Style = .{};

    /// Styling for `file` (stdout for reports, stderr for the error line).
    pub fn detect(environ: *const std.process.Environ.Map, io: std.Io, file: std.Io.File) Style {
        const th = theme.Theme.detect(environ);
        const tty = file.isTty(io) catch false;
        return .{ .theme = th, .enabled = tty and th.kind != .plain };
    }

    /// The escape that starts a span, or nothing.
    pub fn on(self: Style, kind: Kind) []const u8 {
        return if (self.enabled) self.theme.paint(kind) else "";
    }

    /// The escape that ends a span, or nothing.
    pub fn off(self: Style) []const u8 {
        return if (self.enabled) theme.reset else "";
    }
};

test "a disabled style emits nothing, an enabled one wraps spans in reset" {
    try std.testing.expectEqualStrings("", Style.none.on(.header));
    try std.testing.expectEqualStrings("", Style.none.off());
    const lit: Style = .{ .theme = .{ .kind = .truecolor }, .enabled = true };
    try std.testing.expectEqualStrings("\x1b[1;38;2;142;192;124m", lit.on(.header));
    try std.testing.expectEqualStrings("\x1b[0m", lit.off());
    // A plain theme (NO_COLOR, dumb terminal) never enables.
    const a = std.testing.allocator;
    var map = std.process.Environ.Map.init(a);
    defer map.deinit();
    try map.put("NO_COLOR", "1");
    try map.put("COLORTERM", "truecolor");
    try std.testing.expect(!Style.detect(&map, std.testing.io, std.Io.File.stdout()).enabled);
}
