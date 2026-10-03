//! The server's request log: one line per response on the server's stdout,
//! coloured when the terminal advertises colour (`tui/style.zig`): local
//! time, method, path, status, latency, size, and the handler's note.
//! Connection tasks log concurrently; a line is formatted on the caller's
//! stack and written whole under the lock, so lines never interleave.
const std = @import("std");
const builtin = @import("builtin");
const style = @import("../tui/style.zig");

pub const Entry = struct {
    /// Null when the request never parsed (a refused connection, a bad head).
    method: ?std.http.Method,
    path: []const u8,
    status: std.http.Status,
    duration_ns: u64,
    bytes_out: usize,
    /// What the handler did, or the error's code and message; may be empty.
    note: []const u8 = "",
};

pub const Log = struct {
    out: *std.Io.Writer,
    sty: style.Style,
    mutex: std.Io.Mutex = .init,
    /// Local time's offset from UTC, read once at start.
    offset_s: i64,

    pub fn init(out: *std.Io.Writer, sty: style.Style, io: std.Io) Log {
        const now: i64 = @intCast(@divFloor(std.Io.Clock.real.now(io).nanoseconds, std.time.ns_per_s));
        return .{ .out = out, .sty = sty, .offset_s = utcOffset(now) };
    }

    pub fn request(self: *Log, io: std.Io, entry: Entry) void {
        var buffer: [1024]u8 = undefined;
        var line: std.Io.Writer = .fixed(&buffer);
        const ns = std.Io.Clock.real.now(io).nanoseconds;
        format(&line, self.sty, entry, @intCast(@divFloor(ns, std.time.ns_per_ms) + self.offset_s * std.time.ms_per_s)) catch {
            // A note too long for the line: end it there.
            line.end = buffer.len - 4;
            line.writeAll("…\n") catch {};
        };
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.out.writeAll(line.buffered()) catch return;
        self.out.flush() catch {};
    }
};

/// `HH:MM:SS.mmm METHOD path status latency size  note`, then a newline;
/// `local_ms` is milliseconds since the epoch in local time.
pub fn format(w: *std.Io.Writer, sty: style.Style, e: Entry, local_ms: u64) std.Io.Writer.Error!void {
    const off = sty.off();
    const day_ms = local_ms % std.time.ms_per_day;
    try w.print("{s}{d:0>2}:{d:0>2}:{d:0>2}.{d:0>3}{s} ", .{ sty.on(.dim), day_ms / std.time.ms_per_hour, day_ms / std.time.ms_per_min % 60, day_ms / std.time.ms_per_s % 60, day_ms % 1000, off });
    const method = if (e.method) |m| @tagName(m) else "-";
    try w.print("{s}{s: <6}{s} {s}{s: <16}{s} ", .{ sty.on(.bold), method, off, sty.on(.code), e.path, off });
    const code = @backingInt(e.status);
    const status_kind: style.Kind = if (code < 300) .success else if (code < 500) .warning else .error_text;
    try w.print("{s}{d}{s} ", .{ sty.on(status_kind), code, off });
    var latency: [16]u8 = undefined;
    var size: [16]u8 = undefined;
    try w.print("{s}{s: >9}{s} {s}{s: >8}{s}", .{ sty.on(.number), formatDuration(&latency, e.duration_ns), off, sty.on(.dim), formatSize(&size, e.bytes_out), off });
    if (e.note.len > 0) try w.print("  {s}{s}{s}", .{ sty.on(if (code < 400) .dim else status_kind), e.note, off });
    try w.writeByte('\n');
}

fn formatDuration(buffer: []u8, ns: u64) []const u8 {
    const ms = @as(f64, @floatFromInt(ns)) / std.time.ns_per_ms;
    return (if (ms < 1)
        std.fmt.bufPrint(buffer, "{d:.3} ms", .{ms})
    else if (ms < 1000)
        std.fmt.bufPrint(buffer, "{d:.1} ms", .{ms})
    else
        std.fmt.bufPrint(buffer, "{d:.2} s", .{ms / 1000})) catch "?";
}

fn formatSize(buffer: []u8, bytes: usize) []const u8 {
    const b: f64 = @floatFromInt(bytes);
    return (if (bytes < 1024)
        std.fmt.bufPrint(buffer, "{d} B", .{bytes})
    else if (bytes < 1024 * 1024)
        std.fmt.bufPrint(buffer, "{d:.1} KB", .{b / 1024})
    else
        std.fmt.bufPrint(buffer, "{d:.1} MB", .{b / (1024 * 1024)})) catch "?";
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

/// The local zone's offset at `now` (seconds since the epoch); UTC where
/// libc's zone database is not linked.
fn utcOffset(now: i64) i64 {
    if (!builtin.target.os.tag.isDarwin()) return 0;
    var tm: Tm = undefined;
    const t: c_long = @intCast(now);
    return if (localtime_r(&t, &tm) != null) tm.gmtoff else 0;
}

test "a line: time, method, path, status, latency, size, note" {
    var buffer: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buffer);
    // 13:04:05.678 on some day.
    const ms: u64 = 20_000 * std.time.ms_per_day + 13 * std.time.ms_per_hour + 4 * std.time.ms_per_min + 5_678;
    try format(&w, .none, .{ .method = .POST, .path = "/v1/decisions", .status = .ok, .duration_ns = 14_210_000, .bytes_out = 612, .note = "laya · 1 state × 2 questions" }, ms);
    try std.testing.expectEqualStrings("13:04:05.678 POST   /v1/decisions    200   14.2 ms    612 B  laya · 1 state × 2 questions\n", w.buffered());
    w = .fixed(&buffer);
    try format(&w, .none, .{ .method = null, .path = "-", .status = @fromBackingInt(529), .duration_ns = 42_000, .bytes_out = 3 * 1024 * 1024 }, 0);
    try std.testing.expectEqualStrings("00:00:00.000 -      -                529  0.042 ms   3.0 MB\n", w.buffered());
}
