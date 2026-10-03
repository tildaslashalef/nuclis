//! Compile-time text helpers for the executable.

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
