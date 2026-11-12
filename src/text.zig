//! Text helpers for the executable: comptime repetition, and the
//! sanitiser for outside bytes bound for a log or a terminal.

const std = @import("std");

/// `s` repeated `n` times as a comptime string, sentinel-terminated like a
/// literal. Zig has no repetition operator, and `@splat` repeats one byte.
/// Each copy is one comptime branch; a caller looping over it raises its quota.
pub inline fn repeat(comptime s: []const u8, comptime n: usize) *const [s.len * n:0]u8 {
    comptime {
        var out: [s.len * n:0]u8 = undefined;
        for (0..n) |i| @memcpy(out[i * s.len ..][0..s.len], s);
        const final = out;
        return &final;
    }
}

test repeat {
    try std.testing.expectEqualStrings("", repeat("ab", 0));
    try std.testing.expectEqualStrings("ababab", repeat("ab", 3));
    try std.testing.expectEqualStrings("──", repeat("─", 2));
    try std.testing.expectEqual(0, repeat("x", 4)[4]);
}

/// `text` copied into `out` with every byte that could steer a terminal or
/// break a JSON string made `?`: C0 and C1 controls, DEL, and invalid
/// UTF-8. Stops before a sequence that would not fit whole.
pub fn printable(out: []u8, text: []const u8) []const u8 {
    var n: usize = 0;
    var i: usize = 0;
    while (i < text.len) {
        const c = text[i];
        var take: []const u8 = "?";
        var step: usize = 1;
        if (c >= 0x20 and c < 0x7f) {
            take = text[i..][0..1];
        } else if (c >= 0x80) {
            if (std.unicode.utf8ByteSequenceLength(c)) |len| {
                if (i + len <= text.len) {
                    if (std.unicode.utf8Decode(text[i..][0..len])) |cp| {
                        step = len;
                        if (cp >= 0xa0) take = text[i..][0..len];
                    } else |_| {}
                }
            } else |_| {}
        }
        if (n + take.len > out.len) break;
        @memcpy(out[n..][0..take.len], take);
        n += take.len;
        i += step;
    }
    return out[0..n];
}

test printable {
    var out: [16]u8 = undefined;
    try std.testing.expectEqualStrings("/v1/models", printable(&out, "/v1/models"));
    try std.testing.expectEqualStrings("a?[2Jb?", printable(&out, "a\x1b[2Jb\x7f"));
    try std.testing.expectEqualStrings("é?x", printable(&out, "é\xc2\x9bx"));
    try std.testing.expectEqualStrings("??x", printable(&out, "\xff\xc3x"));
    try std.testing.expectEqualStrings("abc", printable(out[0..4], "abcé"));
    try std.testing.expectEqualStrings("", printable(&out, ""));
}
