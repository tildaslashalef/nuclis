//! One live step on stderr, for a command that works for a while before its
//! result (a benchmark's runs, an evaluation's windows). On a terminal the
//! step is one line redrawn in place, erased or turned into a final status
//! line at the end; off a terminal every update is its own timestamped line
//! (`14:32:01 ▸ run 2 of 4`), so a log keeps the history. Writing errors
//! are dropped: progress must never fail the command it reports on.
const std = @import("std");
const Ui = @import("Ui.zig");
const text = @import("text.zig");
const style = @import("style.zig");
const blocks = @import("blocks.zig");
const values = @import("values.zig");

const Step = @This();

io: std.Io,
/// Settings for stderr; `width` null means not a terminal.
ui: Ui,
offset_s: i64,
frame: usize = 0,
/// A redrawn line is on screen and not yet ended.
open: bool = false,

pub fn init(io: std.Io, environ: *const std.process.Environ.Map) Step {
    return .{
        .io = io,
        .ui = Ui.detect(environ, io, std.Io.File.stderr()),
        .offset_s = values.utcOffset(std.Io.Timestamp.now(io, .real).toSeconds()),
    };
}

/// Whether updates redraw one line (a terminal) rather than append lines.
pub fn live(self: Step) bool {
    return self.ui.width != null;
}

/// The step's current state: redrawn in place on a terminal, a new line off one.
pub fn update(self: *Step, comptime fmt: []const u8, args: anytype) void {
    var line_buffer: [512]u8 = undefined;
    const line = std.fmt.bufPrint(&line_buffer, fmt, args) catch line_buffer[0..];
    var buffer: [1024]u8 = undefined;
    var stderr = std.Io.File.stderr().writerStreaming(self.io, &buffer);
    const w = &stderr.interface;
    self.write(w, line) catch {};
    w.flush() catch {};
}

fn write(self: *Step, w: *std.Io.Writer, line: []const u8) !void {
    const g = self.ui.glyphs;
    if (self.ui.width) |cols| {
        const glyph = g.spinner[self.frame % g.spinner.len];
        self.frame +%= 1;
        try w.print("\r\x1b[K{s}{s}{s} ", .{ self.ui.on(.id), glyph, self.ui.off(.id) });
        // The last column stays empty: writing it would wrap the line.
        try text.writeFit(w, line, @min(text.width(line), cols -| (text.width(glyph) + 2)), .left, g.ellipsis, false);
        self.open = true;
        return;
    }
    const epoch = std.Io.Timestamp.now(self.io, .real).toSeconds();
    try w.print("{f} {s} {s}\n", .{ values.clock(epoch, self.offset_s), g.step, line });
}

/// Erases the redrawn line, so what follows on the terminal starts clean.
pub fn clear(self: *Step) void {
    if (!self.open) return;
    var buffer: [16]u8 = undefined;
    var stderr = std.Io.File.stderr().writerStreaming(self.io, &buffer);
    stderr.interface.writeAll("\r\x1b[K") catch {};
    stderr.interface.flush() catch {};
    self.open = false;
}

/// Ends the step with a status line (`✓ subject  detail`) on a terminal and
/// off one alike.
pub fn done(self: *Step, s: style.Status, subject: []const u8, detail: []const u8) void {
    self.clear();
    var buffer: [1024]u8 = undefined;
    var stderr = std.Io.File.stderr().writerStreaming(self.io, &buffer);
    blocks.result(&stderr.interface, self.ui, s, subject, detail) catch {};
    stderr.interface.flush() catch {};
}

test "off a terminal every update is a timestamped line" {
    var step: Step = .{ .io = std.testing.io, .ui = .plain, .offset_s = 0 };
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try step.write(&out.writer, "run 1 of 3");
    try std.testing.expect(std.mem.endsWith(u8, out.written(), " ▸ run 1 of 3\n"));
    try std.testing.expectEqual(@as(usize, "00:00:00".len), std.mem.indexOfScalar(u8, out.written(), ' ').?);
    try std.testing.expect(!step.open);
}

test "on a terminal the line is redrawn and fits the width" {
    var step: Step = .{ .io = std.testing.io, .ui = .{ .width = 12 }, .offset_s = 0 };
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try step.write(&out.writer, "run 1 of 3, warm-up");
    try std.testing.expect(std.mem.startsWith(u8, out.written(), "\r\x1b[K"));
    try std.testing.expect(step.open);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "warm-up") == null);
}
