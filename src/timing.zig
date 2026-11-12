//! Stage timings for the commands' and the server's reports: one awake-clock
//! lap per stage, reported in milliseconds.
const std = @import("std");

/// Nanoseconds on the awake clock since `started`, which moves to now.
pub fn lap(io: std.Io, started: *std.Io.Timestamp) u64 {
    const now = std.Io.Clock.awake.now(io);
    defer started.* = now;
    return @intCast(@max(0, started.durationTo(now).toNanoseconds())); // checked: clamped non-negative; one stage, below 584 years
}

pub fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / std.time.ns_per_ms;
}

/// `x` to one decimal, as the JSON reports write milliseconds.
pub fn round1(x: f64) f64 {
    return @round(x * 10) / 10;
}

test "milliseconds, and to one decimal" {
    try std.testing.expectEqual(@as(f64, 1.5), ms(1_500_000));
    try std.testing.expectEqual(@as(f64, 1.3), round1(ms(1_250_001)));
    try std.testing.expectEqual(@as(f64, 0), round1(ms(49_999)));
}

test "a lap measures from the last and moves the start" {
    const io = std.testing.io;
    var started = std.Io.Clock.awake.now(io);
    const before = started;
    _ = lap(io, &started);
    try std.testing.expect(before.durationTo(started).toNanoseconds() >= 0);
}
