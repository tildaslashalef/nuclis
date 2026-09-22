//! Pure key decoding for the chat's raw-mode input. Two encodings arrive:
//!
//! - Legacy bytes: printable text, control bytes (Ctrl-C = 3), `\r` for Enter,
//!   and `ESC [ ...` sequences for arrows and friends, which we skip.
//! - Kitty keyboard protocol (`CSI code ; modifiers u`), requested by the
//!   terminal lease with `CSI > 1 u`. Terminals that support it report
//!   Shift+Enter as `CSI 13;2u`, which is otherwise indistinguishable from
//!   Enter; they also report Ctrl+letter as `CSI <letter>;5u`. Terminals
//!   without support ignore the request and keep sending legacy bytes.
//!
//! `next` consumes one key from the front of a buffer. Multi-byte UTF-8 text
//! passes through as `.text` slices; an incomplete sequence at the end of the
//! buffer is reported so the caller can wait for more bytes.
const std = @import("std");

pub const Key = union(enum) {
    text: []const u8,
    enter,
    /// Alt-Enter: legacy `ESC CR`/`ESC LF`, or the kitty/modifyOtherKeys
    /// form of Enter with the alt bit. Submits when idle; during a turn it
    /// queues the text for after the turn where Enter steers.
    alt_enter,
    newline,
    backspace,
    delete,
    left,
    right,
    up,
    down,
    home,
    end,
    tab,
    ctrl: u8,
    /// Bracketed paste markers (`CSI 200~` / `CSI 201~`): between them the
    /// terminal delivers pasted text verbatim, newlines included.
    paste_begin,
    paste_end,
    /// Focus reports (`CSI I` / `CSI O`), requested by the terminal lease.
    focus_in,
    focus_out,
    /// A sequence we recognize but do not use (function keys, modifiers).
    ignored,
};

pub const Decoded = struct { key: Key, consumed: usize };

/// Decodes the first key in `bytes`; returns null when the buffer holds an
/// incomplete sequence (caller should read more). `bytes` must be nonempty.
pub fn next(bytes: []const u8) ?Decoded {
    const b = bytes[0];
    if (b == 27) return escape(bytes);
    switch (b) {
        13 => return .{ .key = .enter, .consumed = 1 },
        10 => return .{ .key = .newline, .consumed = 1 },
        9 => return .{ .key = .tab, .consumed = 1 },
        127, 8 => return .{ .key = .backspace, .consumed = 1 },
        1...7, 11, 12, 14...26 => return .{ .key = .{ .ctrl = b + 'a' - 1 }, .consumed = 1 },
        28...31 => return .{ .key = .ignored, .consumed = 1 },
        else => {},
    }
    const len = std.unicode.utf8ByteSequenceLength(b) catch return .{ .key = .ignored, .consumed = 1 };
    if (len > bytes.len) return null;
    if (!std.unicode.utf8ValidateSlice(bytes[0..len])) return .{ .key = .ignored, .consumed = 1 };
    return .{ .key = .{ .text = bytes[0..len] }, .consumed = len };
}

fn escape(bytes: []const u8) ?Decoded {
    if (bytes.len < 2) return null; // lone ESC or split sequence: wait
    if (bytes[1] == '\r' or bytes[1] == '\n') return .{ .key = .alt_enter, .consumed = 2 };
    if (bytes[1] != '[' and bytes[1] != 'O') return .{ .key = .ignored, .consumed = 2 }; // Alt+key
    // CSI: parameters (0x30-0x3f), intermediates (0x20-0x2f), final (0x40-0x7e).
    var i: usize = 2;
    while (i < bytes.len and bytes[i] >= 0x20 and bytes[i] <= 0x3f) i += 1;
    if (i >= bytes.len) return null;
    const final = bytes[i];
    const consumed = i + 1;
    if (bytes[1] == '[' and final == 'I') return .{ .key = .focus_in, .consumed = consumed };
    if (bytes[1] == '[' and final == 'O') return .{ .key = .focus_out, .consumed = consumed };
    if (bytes[1] == '[' and final == 'u') {
        // Kitty: code[:alternates][;modifiers[:event]]u
        var params = std.mem.splitScalar(u8, bytes[2..i], ';');
        const code_field = params.next() orelse return .{ .key = .ignored, .consumed = consumed };
        const code = parseFirst(code_field) orelse return .{ .key = .ignored, .consumed = consumed };
        const modifiers = if (params.next()) |m| (parseFirst(m) orelse 1) -| 1 else 0;
        return .{ .key = kitty(code, modifiers), .consumed = consumed };
    }
    if (bytes[1] == '[' and final == '~') {
        // Legacy tilde codes (CSI 3~ = Delete, CSI 1~/7~ = Home, CSI 4~/8~ =
        // End) and xterm modifyOtherKeys (27;modifiers;code~).
        var params = std.mem.splitScalar(u8, bytes[2..i], ';');
        const first = params.next() orelse "";
        if (std.mem.eql(u8, first, "27")) {
            const modifiers = (parseFirst(params.next() orelse "") orelse 1) -| 1;
            if (parseFirst(params.next() orelse "")) |code| return .{ .key = kitty(code, modifiers), .consumed = consumed };
        } else if (params.next() == null) {
            if (parseFirst(first)) |code| return .{ .key = tilde(code), .consumed = consumed };
        }
    }
    if (bytes[1] == '[' or bytes[1] == 'O') {
        // Legacy cursor keys: CSI 1;modsA/B/C/D (arrows), H (home), F (end).
        // Only unmodified presses are used; modifier combos stay unused.
        const key: ?Key = switch (final) {
            'A' => .up,
            'B' => .down,
            'C' => .right,
            'D' => .left,
            'H' => .home,
            'F' => .end,
            else => null,
        };
        if (key) |k| {
            // Parameters are `[code][;modifiers]`: plain only when the code is
            // absent or 1 and the modifier field is missing or 1 (none).
            var ps = std.mem.splitScalar(u8, bytes[2..i], ';');
            const code_p = ps.next() orelse "";
            const mods: u32 = if (ps.next()) |m| (parseFirst(m) orelse 1) else 1;
            if ((code_p.len == 0 or std.mem.eql(u8, code_p, "1")) and mods == 1) {
                return .{ .key = k, .consumed = consumed };
            }
        }
    }
    return .{ .key = .ignored, .consumed = consumed };
}

const shift: u32 = 1;
const alt: u32 = 2;
const ctrl: u32 = 4;

fn kitty(code: u32, modifiers: u32) Key {
    if (modifiers == alt and code == 13) return .alt_enter;
    const plain = modifiers & ~(shift | ctrl) == 0;
    if (!plain) return .ignored; // alt/super/hyper/meta combinations
    if (modifiers & ctrl != 0) {
        if (code >= 'a' and code <= 'z') return .{ .ctrl = @intCast(code) };
        if (code >= 'A' and code <= 'Z') return .{ .ctrl = @intCast(code + 32) };
        return .ignored;
    }
    return switch (code) {
        13 => if (modifiers & shift != 0) .newline else .enter,
        9 => .tab,
        127 => .backspace,
        27 => .ignored,
        3 => .delete,
        // Kitty functional codes for arrows (63234-63237); plain presses
        // usually arrive as legacy CSI A/B/C/D instead.
        63234 => .left,
        63235 => .right,
        63236 => .up,
        63237 => .down,
        else => .ignored, // text arrives as legacy bytes; other keys are unused
    };
}

/// Legacy `CSI n~` codes for unmodified presses.
fn tilde(code: u32) Key {
    return switch (code) {
        3 => .delete,
        1, 7 => .home,
        4, 8 => .end,
        200 => .paste_begin,
        201 => .paste_end,
        else => .ignored,
    };
}

fn parseFirst(field: []const u8) ?u32 {
    var parts = std.mem.splitScalar(u8, field, ':');
    return std.fmt.parseInt(u32, parts.next() orelse return null, 10) catch null;
}

test "legacy bytes decode to text, enter, controls" {
    try std.testing.expectEqual(Key.enter, next("\r").?.key);
    try std.testing.expectEqual(Key.newline, next("\n").?.key);
    try std.testing.expectEqual(Key.tab, next("\t").?.key);
    try std.testing.expectEqual(Key.backspace, next("\x7f").?.key);
    try std.testing.expectEqual(Key{ .ctrl = 'c' }, next("\x03").?.key);
    try std.testing.expectEqual(Key{ .ctrl = 't' }, next("\x14").?.key);
    const d = next("中x").?;
    try std.testing.expectEqualStrings("中", d.key.text);
    try std.testing.expectEqual(@as(usize, 3), d.consumed);
    try std.testing.expect(next("\xe4\xb8") == null); // split UTF-8: wait for more
}

test "kitty protocol distinguishes shift+enter and encodes ctrl letters" {
    try std.testing.expectEqual(Key.newline, next("\x1b[13;2u").?.key);
    try std.testing.expectEqual(Key.enter, next("\x1b[13u").?.key);
    try std.testing.expectEqual(Key.enter, next("\x1b[13;1u").?.key);
    try std.testing.expectEqual(Key{ .ctrl = 'c' }, next("\x1b[99;5u").?.key);
    try std.testing.expectEqual(Key{ .ctrl = 'j' }, next("\x1b[106;5u").?.key);
    try std.testing.expectEqual(Key{ .ctrl = 'n' }, next("\x1b[110;5:1u").?.key); // with event type
    try std.testing.expectEqual(Key.alt_enter, next("\x1b[13;3u").?.key); // alt+enter (kitty)
    try std.testing.expectEqual(Key.alt_enter, next("\x1b[27;3;13~").?.key); // alt+enter (modifyOtherKeys)
    try std.testing.expectEqual(Key.alt_enter, next("\x1b\r").?.key); // alt+enter (legacy)
    try std.testing.expectEqual(@as(usize, 2), next("\x1b\r").?.consumed);
    try std.testing.expectEqual(Key.ignored, next("\x1b[13;7u").?.key); // ctrl+alt+enter stays unused
    try std.testing.expectEqual(Key.newline, next("\x1b[27;2;13~").?.key); // xterm modifyOtherKeys
}

test "focus reports decode from CSI I and CSI O" {
    try std.testing.expectEqual(Key.focus_in, next("\x1b[I").?.key);
    try std.testing.expectEqual(@as(usize, 3), next("\x1b[I").?.consumed);
    try std.testing.expectEqual(Key.focus_out, next("\x1b[Ox").?.key);
    // SS3 `ESC O A` is arrow up, and other SS3 finals stay ignored.
    try std.testing.expectEqual(Key.up, next("\x1bOA").?.key);
    try std.testing.expectEqual(Key.ignored, next("\x1bOP").?.key);
}

test "cursor keys decode plain presses, modifiers stay unused" {
    try std.testing.expectEqual(Key.up, next("\x1b[Ax").?.key);
    try std.testing.expectEqual(@as(usize, 3), next("\x1b[A").?.consumed);
    try std.testing.expectEqual(Key.left, next("\x1b[1;1D").?.key);
    try std.testing.expectEqual(Key.home, next("\x1b[H").?.key);
    try std.testing.expectEqual(Key.end, next("\x1bOF").?.key);
    try std.testing.expectEqual(Key.delete, next("\x1b[3~").?.key);
    try std.testing.expectEqual(Key.ignored, next("\x1b[3;5~").?.key); // Ctrl+Delete
    try std.testing.expectEqual(Key.ignored, next("\x1b[1;5C").?.key); // Ctrl+Right
    try std.testing.expectEqual(@as(usize, 6), next("\x1b[1;5C").?.consumed);
    try std.testing.expectEqual(@as(usize, 6), next("\x1b[200~").?.consumed);
    try std.testing.expectEqual(Key.paste_begin, next("\x1b[200~").?.key);
    try std.testing.expectEqual(Key.paste_end, next("\x1b[201~text").?.key);
    try std.testing.expectEqual(Key.right, next("\x1b[63235u").?.key);
    try std.testing.expect(next("\x1b") == null);
    try std.testing.expect(next("\x1b[1;5") == null);
    try std.testing.expectEqual(@as(usize, 2), next("\x1bf").?.consumed); // alt-f
}
