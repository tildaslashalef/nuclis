//! How a value reads: counted nouns, sizes, durations, percentages. One
//! formatter each, so every command writes `16.56 GB` and `3m 12s` alike.
//! Sizes are decimal (the Hub's and the documentation's units); JSON keeps
//! exact bytes and seconds.
const std = @import("std");
const builtin = @import("builtin");

/// A missing value in a table or detail view, painted muted.
pub const missing = "--";

/// `1 file`, `3 files`; `plural` for an irregular noun (`entry`/`entries`).
pub const Count = struct {
    n: u64,
    noun: []const u8,
    plural: ?[]const u8,

    pub fn format(c: Count, w: *std.Io.Writer) std.Io.Writer.Error!void {
        if (c.n == 1) return w.print("1 {s}", .{c.noun});
        if (c.plural) |p| return w.print("{d} {s}", .{ c.n, p });
        try w.print("{d} {s}s", .{ c.n, c.noun });
    }
};

pub fn count(n: anytype, noun: []const u8) Count {
    return .{ .n = @intCast(n), .noun = noun, .plural = null };
}

pub fn countIrregular(n: anytype, noun: []const u8, plural: []const u8) Count {
    return .{ .n = @intCast(n), .noun = noun, .plural = plural };
}

/// Decimal units, two decimals from a gigabyte up, one below (`16.46 GB`,
/// `931.1 MB`, `12.0 KB`), exact below a kilobyte (`512 B`).
pub const Size = struct {
    bytes: u64,

    pub fn format(s: Size, w: *std.Io.Writer) std.Io.Writer.Error!void {
        const b: f64 = @floatFromInt(s.bytes);
        if (s.bytes >= 1_000_000_000) return w.print("{d:.2} GB", .{b / 1e9});
        if (s.bytes >= 1_000_000) return w.print("{d:.1} MB", .{b / 1e6});
        if (s.bytes >= 1_000) return w.print("{d:.1} KB", .{b / 1e3});
        try w.print("{d} B", .{s.bytes});
    }
};

pub fn size(bytes: u64) Size {
    return .{ .bytes = bytes };
}

/// Below a second in milliseconds (`812 ms`, `14.2 ms` under 100), then
/// the two largest units (`42.1 s` under a minute, `5m 03s`, `3h 12m`).
pub const Duration = struct {
    seconds: f64,

    pub fn format(d: Duration, w: *std.Io.Writer) std.Io.Writer.Error!void {
        const s = @max(d.seconds, 0);
        if (s < 0.1) return w.print("{d:.1} ms", .{s * 1000});
        if (s < 1) return w.print("{d:.0} ms", .{s * 1000});
        if (s < 60) return w.print("{d:.1} s", .{s});
        const whole: u64 = @intFromFloat(@round(s));
        if (whole < 3600) return w.print("{d}m {d:0>2}s", .{ whole / 60, whole % 60 });
        if (whole < 86400) return w.print("{d}h {d:0>2}m", .{ whole / 3600, whole % 3600 / 60 });
        try w.print("{d}d {d}h", .{ whole / 86400, whole % 86400 / 3600 });
    }
};

pub fn duration(seconds: f64) Duration {
    return .{ .seconds = seconds };
}

/// A fraction (0.425) as `42.5%`.
pub const Percent = struct {
    fraction: f64,

    pub fn format(p: Percent, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("{d:.1}%", .{p.fraction * 100});
    }
};

pub fn percent(fraction: f64) Percent {
    return .{ .fraction = fraction };
}

/// A count of anything with thousands separators: `65,536`.
pub const Grouped = struct {
    n: u64,

    pub fn format(g: Grouped, w: *std.Io.Writer) std.Io.Writer.Error!void {
        var digits: [20]u8 = undefined;
        const s = std.fmt.bufPrint(&digits, "{d}", .{g.n}) catch unreachable;
        for (s, 0..) |c, i| {
            if (i > 0 and (s.len - i) % 3 == 0) try w.writeByte(',');
            try w.writeByte(c);
        }
    }
};

pub fn grouped(n: anytype) Grouped {
    return .{ .n = @intCast(n) };
}

/// `14:32:01`: the time of day of `epoch_s` (seconds since the epoch) in
/// the zone `offset_s` from UTC (`utcOffset`).
pub const Clock = struct {
    epoch_s: i64,
    offset_s: i64,

    pub fn format(c: Clock, w: *std.Io.Writer) std.Io.Writer.Error!void {
        const day: u64 = @intCast(@mod(c.epoch_s + c.offset_s, 86400));
        try w.print("{d:0>2}:{d:0>2}:{d:0>2}", .{ day / 3600, day % 3600 / 60, day % 60 });
    }
};

pub fn clock(epoch_s: i64, offset_s: i64) Clock {
    return .{ .epoch_s = epoch_s, .offset_s = offset_s };
}

const Tm = extern struct {
    sec: c_int,
    min: c_int,
    hour: c_int,
    mday: c_int,
    mon: c_int,
    year: c_int,
    wday: c_int,
    yday: c_int,
    isdst: c_int,
    gmtoff: c_long,
    zone: ?[*:0]const u8,
};
extern "c" fn localtime_r(time: *const c_long, result: *Tm) ?*Tm;

/// The local zone's offset from UTC at `now` (seconds since the epoch);
/// zero where libc's zone database is not linked.
pub fn utcOffset(now: i64) i64 {
    if (!builtin.target.os.tag.isDarwin()) return 0;
    var tm: Tm = undefined;
    const t: c_long = @intCast(now); // checked: the system clock, not a client's
    return if (localtime_r(&t, &tm) != null) tm.gmtoff else 0;
}

const testing = std.testing;

fn expectFmt(expected: []const u8, v: anytype) !void {
    var buf: [64]u8 = undefined;
    try testing.expectEqualStrings(expected, try std.fmt.bufPrint(&buf, "{f}", .{v}));
}

test "values read the same everywhere" {
    try expectFmt("1 file", count(1, "file"));
    try expectFmt("3 files", count(3, "file"));
    try expectFmt("0 files", count(0, "file"));
    try expectFmt("2 entries", countIrregular(2, "entry", "entries"));
    try expectFmt("16.46 GB", size(16464440224));
    try expectFmt("931.1 MB", size(931_100_000));
    try expectFmt("12.0 KB", size(12_000));
    try expectFmt("512 B", size(512));
    try expectFmt("14.2 ms", duration(0.0142));
    try expectFmt("812 ms", duration(0.812));
    try expectFmt("42.1 s", duration(42.1));
    try expectFmt("5m 03s", duration(303));
    try expectFmt("3h 12m", duration(3 * 3600 + 12 * 60 + 5));
    try expectFmt("2d 4h", duration(2 * 86400 + 4 * 3600));
    try expectFmt("42.5%", percent(0.425));
    try expectFmt("65,536", grouped(65536));
    try expectFmt("512", grouped(512));
    try expectFmt("1,000,000", grouped(1_000_000));
}
