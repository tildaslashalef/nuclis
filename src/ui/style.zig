//! The kit's roles, glyphs, and status words. Roles are painted through the
//! agent's theme (`tui/theme.zig`, its `ui_*` styles), so colour depth,
//! `NO_COLOR`, and the ASCII fallback follow one detection.
const std = @import("std");
const theme = @import("../tui/theme.zig");

pub const Role = enum {
    /// The terminal's own foreground: no escape at all.
    text,
    muted,
    border,
    title,
    id,
    ok,
    warn,
    fail,
    changed,
    emphasis,
    high,

    pub fn kind(role: Role) ?theme.Style {
        return switch (role) {
            .text => null,
            .muted => .ui_muted,
            .border => .ui_border,
            .title => .ui_title,
            .id => .ui_id,
            .ok => .ui_ok,
            .warn => .ui_warn,
            .fail => .ui_fail,
            .changed => .ui_changed,
            .emphasis => .ui_emphasis,
            .high => .ui_high,
        };
    }
};

pub const Glyphs = struct {
    running: []const u8,
    stopped: []const u8,
    ok: []const u8,
    warn: []const u8,
    fail: []const u8,
    not_checked: []const u8,
    unknown: []const u8,
    ellipsis: []const u8,
    bar_full: []const u8,
    bar_empty: []const u8,
    arrow: []const u8,
    dot: []const u8,
    /// A step starting, in a plain progress line.
    step: []const u8,
    rule: []const u8,
    /// nuclis's mark, leading the header line.
    mark: []const u8,
    /// The spinner of a running step, one frame per entry.
    spinner: []const []const u8,
};

pub const unicode: Glyphs = .{
    .running = "●",
    .stopped = "○",
    .ok = "✓",
    .warn = "▲",
    .fail = "✗",
    .not_checked = "–",
    .unknown = "?",
    .ellipsis = "…",
    .bar_full = "█",
    .bar_empty = "░",
    .arrow = "→",
    .dot = "·",
    .step = "▸",
    .rule = "─",
    .mark = "◆",
    .spinner = &.{ "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏" },
};

pub const ascii: Glyphs = .{
    .running = "*",
    .stopped = "o",
    .ok = "+",
    .warn = "!",
    .fail = "x",
    .not_checked = "-",
    .unknown = "?",
    .ellipsis = "...",
    .bar_full = "#",
    .bar_empty = ".",
    .arrow = "->",
    .dot = "-",
    .step = ">",
    .rule = "-",
    .mark = "nuclis",
    .spinner = &.{ "|", "/", "-", "\\" },
};

/// A state shown as a glyph and a word in the state's colour. The word is
/// the caller's (`✓ installed`, `▲ unverified`); `word` is the default.
pub const Status = enum {
    running,
    stopped,
    ok,
    warning,
    failed,
    not_checked,
    unknown,

    pub fn word(s: Status) []const u8 {
        return switch (s) {
            .not_checked => "not checked",
            else => @tagName(s),
        };
    }

    pub fn glyph(s: Status, g: *const Glyphs) []const u8 {
        return switch (s) {
            .running => g.running,
            .stopped => g.stopped,
            .ok => g.ok,
            .warning => g.warn,
            .failed => g.fail,
            .not_checked => g.not_checked,
            .unknown => g.unknown,
        };
    }

    pub fn role(s: Status) Role {
        return switch (s) {
            .running, .ok => .ok,
            .warning => .warn,
            .failed => .fail,
            .stopped, .not_checked, .unknown => .muted,
        };
    }
};

test "every role but text has a theme style, and both glyph sets fill every field" {
    for (std.enums.values(Role)) |r| try std.testing.expectEqual(r == .text, r.kind() == null);
    const info = @typeInfo(Glyphs).@"struct";
    inline for (info.field_names, info.field_types) |name, T| {
        if (T == []const u8) {
            try std.testing.expect(@field(unicode, name).len > 0);
            try std.testing.expect(@field(ascii, name).len > 0);
            for (@field(ascii, name)) |b| try std.testing.expect(b < 0x80);
        }
    }
    try std.testing.expectEqualStrings("✓", Status.ok.glyph(&unicode));
    try std.testing.expectEqualStrings("not checked", Status.not_checked.word());
}
