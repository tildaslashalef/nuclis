//! The server's request log: one line per response on the server's stdout,
//! in the output kit's roles (`ui.zig`): local time, method, path, status,
//! latency, size, and the handler's note.
//! Connection tasks log concurrently; a line is formatted on the caller's
//! stack and written whole under the lock, so lines never interleave.
const std = @import("std");
const kit = @import("../ui.zig");
const printable = @import("../text.zig").printable;
const values = @import("../ui/values.zig");

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
    ui: kit.Ui,
    mutex: std.Io.Mutex = .init,
    /// Local time's offset from UTC, read once at start.
    offset_s: i64,

    pub fn init(out: *std.Io.Writer, ui: kit.Ui, io: std.Io) Log {
        const now: i64 = @intCast(@divFloor(std.Io.Clock.real.now(io).nanoseconds, std.time.ns_per_s)); // checked: the system clock, not a client's
        return .{ .out = out, .ui = ui, .offset_s = values.utcOffset(now) };
    }

    pub fn request(self: *Log, io: std.Io, entry: Entry) void {
        var buffer: [1024]u8 = undefined;
        var line: std.Io.Writer = .fixed(&buffer);
        const ns = std.Io.Clock.real.now(io).nanoseconds;
        format(&line, self.ui, entry, @intCast(@divFloor(ns, std.time.ns_per_ms) + self.offset_s * std.time.ms_per_s)) catch { // checked: the system clock, not a client's
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
pub fn format(w: *std.Io.Writer, ui: kit.Ui, e: Entry, local_ms: u64) std.Io.Writer.Error!void {
    const day_ms = local_ms % std.time.ms_per_day;
    try w.print("{s}{d:0>2}:{d:0>2}:{d:0>2}.{d:0>3}{s} ", .{ ui.on(.muted), day_ms / std.time.ms_per_hour, day_ms / std.time.ms_per_min % 60, day_ms / std.time.ms_per_s % 60, day_ms % 1000, ui.off(.muted) });
    const method = if (e.method) |m| @tagName(m) else "-";
    // A client names the path and may put terminal escapes in it.
    var path_buffer: [512]u8 = undefined;
    try w.print("{s: <6} {s}{s: <16}{s} ", .{ method, ui.on(.id), printable(&path_buffer, e.path), ui.off(.id) });
    const code = @backingInt(e.status);
    const role: kit.style.Role = if (code < 300) .ok else if (code < 500) .warn else .fail;
    try w.print("{s}{d}{s} ", .{ ui.on(role), code, ui.off(role) });
    var latency: [16]u8 = undefined;
    var size: [16]u8 = undefined;
    // Below a millisecond the kit's one decimal would read 0.0 ms.
    const took = if (e.duration_ns < std.time.ns_per_ms)
        std.fmt.bufPrint(&latency, "{d:.2} ms", .{@as(f64, @floatFromInt(e.duration_ns)) / std.time.ns_per_ms}) catch "?"
    else
        std.fmt.bufPrint(&latency, "{f}", .{values.duration(@as(f64, @floatFromInt(e.duration_ns)) / std.time.ns_per_s)}) catch "?";
    try w.print("{s: >9} {s}{s: >8}{s}", .{ took, ui.on(.muted), std.fmt.bufPrint(&size, "{f}", .{values.size(e.bytes_out)}) catch "?", ui.off(.muted) });
    var note_buffer: [512]u8 = undefined;
    const note_role: kit.style.Role = if (code < 400) .muted else role;
    if (e.note.len > 0) try w.print("  {s}{s}{s}", .{ ui.on(note_role), printable(&note_buffer, e.note), ui.off(note_role) });
    try w.writeByte('\n');
}

test "a line: time, method, path, status, latency, size, note" {
    var buffer: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buffer);
    // 13:04:05.678 on some day.
    const ms: u64 = 20_000 * std.time.ms_per_day + 13 * std.time.ms_per_hour + 4 * std.time.ms_per_min + 5_678;
    try format(&w, .plain, .{ .method = .POST, .path = "/v1/decisions", .status = .ok, .duration_ns = 14_210_000, .bytes_out = 612, .note = "laya · 1 state × 2 questions" }, ms);
    try std.testing.expectEqualStrings("13:04:05.678 POST   /v1/decisions    200   14.2 ms    612 B  laya · 1 state × 2 questions\n", w.buffered());
    w = .fixed(&buffer);
    try format(&w, .plain, .{ .method = null, .path = "-", .status = @fromBackingInt(529), .duration_ns = 42_000, .bytes_out = 3 * 1024 * 1024 }, 0);
    try std.testing.expectEqualStrings("00:00:00.000 -      -                529   0.04 ms   3.1 MB\n", w.buffered());
    w = .fixed(&buffer);
    try format(&w, .plain, .{ .method = .GET, .path = "/\x1b[2J", .status = .not_found, .duration_ns = 0, .bytes_out = 0, .note = "no \x1b]0;x\x07" }, 0);
    try std.testing.expect(std.mem.indexOfScalar(u8, w.buffered(), 0x1b) == null);
}
