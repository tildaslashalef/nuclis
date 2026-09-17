//! The status bar: one row of measured state, painted as one string.
//!
//! Everything on it is a measurement or a setting, never an estimate dressed
//! up as a measurement — the rule `bench` set and the surface inherits. A rate
//! that was not measured prints `—`, and the prefill countdown is marked `~`
//! because it is extrapolated from the rate measured so far *this turn*, never
//! from a constant.
//!
//! The value is filled from the turn's `status` events (`apply`) plus the
//! settings the agent knows (context window, effort, the word on the left).
//! `paint` is pure: give it a width, a palette, a spinner frame, and the
//! seconds elapsed, and it returns the row. That is what lets the bar's layout
//! be tested without a model, a clock, or a terminal.
//!
//! Zig note. The row is padded to exactly `width` display cells with the
//! escape-aware width (`view.styledWidth`), so the painted background spans
//! the row and the cursor arithmetic in `screen.zig` still adds up. It is
//! truncated rather than wrapped, at the column rather than at a word: a
//! wrapped bar would occupy two rows and break that arithmetic.
const std = @import("std");
const theme = @import("theme.zig");
const view = @import("view.zig");
const event_mod = @import("event.zig");
const Event = event_mod.Event;

/// Padding for the bar's background; terminals clamp width to 400.
const spaces = " " ** 512;

pub const Paint = struct {
    width: usize,
    th: theme.Theme,
    /// Spinner frame, advanced by the caller on every repaint.
    frame: usize = 0,
    /// Seconds since the turn started, at paint time. The caller has the
    /// clock; this module has none.
    elapsed_seconds: f64 = 0,
};

pub const Status = struct {
    /// The word on the left: "ready", "prefill", "generating", or why a turn
    /// stopped.
    message: []const u8 = "ready",
    busy: bool = false,
    phase: event_mod.Phase = .prefill,
    /// Prefill progress, from the engine's beat: how many of the turn's
    /// prompt tokens have been consumed.
    position: usize = 0,
    target: usize = 0,
    rates: event_mod.Rates = .{},
    prompt_tokens: usize = 0,
    generated: usize = 0,
    context_used: usize = 0,
    context_capacity: usize = 0,
    /// Reasoning effort, as its name: the surface does not know the engine's
    /// enum, and does not need to.
    effort: []const u8 = "off",
    /// The model's short name (the registry or catalogue name, else the
    /// artifact's own), last on the bar so a narrow terminal loses it first.
    model: []const u8 = "",
    /// The agent loop's step in progress (1-based) against its budget, while a
    /// turn runs. `budget == 0` means the surface has no loop to report.
    step: usize = 0,
    budget: usize = 0,
    /// The turn had to replay the conversation into a fresh session.
    replayed: bool = false,

    /// Folds a turn event into the bar. Only two kinds say anything about it:
    /// the progress beat and the end of a turn.
    pub fn apply(self: *Status, e: Event) void {
        switch (e) {
            .status => |beat| {
                self.phase = beat.phase;
                self.position = beat.position;
                self.target = beat.target;
                self.rates = beat.rates;
            },
            .turn_end => |end| {
                self.busy = false;
                self.rates = .{};
                self.prompt_tokens = end.stats.prompt_tokens;
                self.generated = end.stats.generated;
                self.replayed = end.stats.replayed;
                if (end.stats.prompt_tokens > 0 and end.stats.prefill_seconds > 0)
                    self.rates.prefill = @as(f64, @floatFromInt(end.stats.prompt_tokens)) / end.stats.prefill_seconds;
                if (end.stats.generated > 1 and end.stats.decode_seconds > 0)
                    self.rates.decode = @as(f64, @floatFromInt(end.stats.generated - 1)) / end.stats.decode_seconds;
            },
            else => {},
        }
    }

    /// Seconds left in the prefill at the rate measured so far, or null
    /// before there is a measurement to extrapolate from.
    pub fn eta(self: Status, elapsed_seconds: f64) ?f64 {
        if (self.position == 0 or self.position >= self.target or elapsed_seconds <= 0) return null;
        const per_token = elapsed_seconds / @as(f64, @floatFromInt(self.position));
        return per_token * @as(f64, @floatFromInt(self.target - self.position));
    }

    pub fn paint(self: Status, a: std.mem.Allocator, options: Paint) ![]const u8 {
        const th = options.th;
        const gl = th.glyphs();
        var w: std.Io.Writer.Allocating = .init(a);
        const spin: []const u8 = if (self.busy) gl.spinner[options.frame % gl.spinner.len] else gl.idle;
        try w.writer.print(" {s} {s}", .{ spin, self.message });
        if (self.busy and self.phase == .prefill and self.target > 0 and self.position < self.target) {
            try w.writer.print(" {s}{d}/{d}{s}", .{ th.paint(.progress), self.position, self.target, theme.fg_default });
            if (self.eta(options.elapsed_seconds)) |seconds_left| try w.writer.print(" ~{d:.0}s", .{seconds_left});
        } else if (self.busy and self.phase == .decode) {
            try w.writer.print(" {s}{d}{s} out, {d:.0}s", .{ th.paint(.progress), self.generated, theme.fg_default, options.elapsed_seconds });
        }
        if (self.busy and self.budget > 0)
            try w.writer.print(" {s} step {d}/{d}", .{ gl.table_bar, self.step, self.budget });
        try w.writer.print(" {s} {s} ctx {s}{d}/{d}{s} {s} {s} in {d} out {d} {s} {s} pp ", .{
            gl.table_bar, gl.context, th.paint(.accent),  self.context_used, self.context_capacity, theme.fg_default,
            gl.table_bar, gl.tokens,  self.prompt_tokens, self.generated,    gl.table_bar,          gl.prefill,
        });
        try rate(&w.writer, self.rates.prefill);
        try w.writer.print(" t/s {s} {s} tg ", .{ gl.table_bar, gl.decode });
        try rate(&w.writer, self.rates.decode);
        try w.writer.print(" t/s {s} {s} think {s}{s}{s}", .{ gl.table_bar, gl.effort, th.paint(.accent), self.effort, theme.fg_default });
        if (self.replayed) try w.writer.print(" {s} replayed", .{gl.table_bar});
        if (self.model.len != 0) try w.writer.print(" {s} {s}", .{ gl.table_bar, self.model });
        const wrapped = try view.wrapStyled(a, w.written(), options.width, .character);
        const pad = options.width -| view.styledWidth(wrapped[0]);
        return std.mem.concat(a, u8, &.{ wrapped[0], spaces[0..@min(pad, spaces.len)] });
    }
};

fn rate(w: *std.Io.Writer, value: ?f64) !void {
    if (value) |v| try w.print("{d:.2}", .{v}) else try w.writeAll("—");
}

// ----- tests -----

const testing = std.testing;

/// The bar as a reader sees it: painted, then stripped of its escape
/// sequences, since the styles are pinned in `theme.zig` and what matters
/// here is the text and its order.
fn painted(a: std.mem.Allocator, status: Status, options: Paint) ![]const u8 {
    const row = try status.paint(a, options);
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < row.len) {
        if (row[i] == 0x1b) {
            i += 1;
            if (i < row.len and row[i] == '[') {
                i += 1;
                while (i < row.len and row[i] >= 0x20 and row[i] <= 0x3f) i += 1;
            }
            if (i < row.len) i += 1;
            continue;
        }
        try out.append(a, row[i]);
        i += 1;
    }
    return out.items;
}

test "an idle bar states the settings and no rate it did not measure" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const status: Status = .{ .context_used = 129, .context_capacity = 4096, .effort = "low" };
    const row = try painted(a, status, .{ .width = 100, .th = .{ .kind = .plain } });
    try testing.expect(std.mem.indexOf(u8, row, "◆ ready") != null);
    try testing.expect(std.mem.indexOf(u8, row, "ctx 129/4096") != null);
    try testing.expect(std.mem.indexOf(u8, row, "pp — t/s") != null);
    try testing.expect(std.mem.indexOf(u8, row, "tg — t/s") != null);
    try testing.expect(std.mem.indexOf(u8, row, "think low") != null);
    var named = status;
    named.model = "hauhau";
    const with_model = try named.paint(a, .{ .width = 120, .th = .{ .kind = .plain }, .frame = 0, .elapsed_seconds = 0 });
    try testing.expect(std.mem.endsWith(u8, std.mem.trimEnd(u8, with_model, " "), " hauhau"));
    try testing.expect(std.mem.indexOf(u8, with_model, "think low") != null);
    try testing.expect(std.mem.indexOf(u8, row, "replayed") == null);
    // The row is exactly as wide as the bar, so its background spans it.
    try testing.expectEqual(@as(usize, 100), view.styledWidth(try status.paint(a, .{ .width = 100, .th = .{ .kind = .plain } })));
}

test "a prefilling bar counts tokens and marks its estimate" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var status: Status = .{ .busy = true, .message = "prefill" };
    status.apply(.{ .status = .{ .phase = .prefill, .position = 768, .target = 1226 } });
    const row = try painted(a, status, .{ .width = 120, .th = .{ .kind = .plain }, .elapsed_seconds = 6 });
    // 768 tokens in 6 s leaves 458 at the same rate: about 4 seconds.
    try testing.expect(std.mem.indexOf(u8, row, "768/1226 ~4s") != null);
    // Before the first chunk there is nothing to extrapolate from.
    var fresh: Status = .{ .busy = true };
    fresh.apply(.{ .status = .{ .phase = .prefill, .position = 0, .target = 1226 } });
    try testing.expect(fresh.eta(3) == null);
    const early = try painted(a, fresh, .{ .width = 120, .th = .{ .kind = .plain }, .elapsed_seconds = 3 });
    try testing.expect(std.mem.indexOf(u8, early, "0/1226") != null);
    try testing.expect(std.mem.indexOf(u8, early, "~") == null);
}

test "a decoding bar shows the live count, and the end freezes the measurements" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var status: Status = .{ .busy = true, .message = "generating", .generated = 12 };
    status.apply(.{ .status = .{ .phase = .decode, .position = 12, .target = 400, .rates = .{ .prefill = 84.26, .decode = 26.1 } } });
    const row = try painted(a, status, .{ .width = 120, .th = .{ .kind = .plain }, .elapsed_seconds = 2.4 });
    try testing.expect(std.mem.indexOf(u8, row, "12 out, 2s") != null);
    try testing.expect(std.mem.indexOf(u8, row, "pp 84.26 t/s") != null);
    try testing.expect(std.mem.indexOf(u8, row, "tg 26.10 t/s") != null);

    status.apply(.{ .turn_end = .{ .stop = .eos, .stats = .{
        .prompt_tokens = 31,
        .generated = 400,
        .prefill_seconds = 0.5,
        .decode_seconds = 10,
        .replayed = true,
    } } });
    const done = try painted(a, status, .{ .width = 120, .th = .{ .kind = .plain } });
    try testing.expect(!status.busy);
    try testing.expect(std.mem.indexOf(u8, done, "in 31 out 400") != null);
    try testing.expect(std.mem.indexOf(u8, done, "pp 62.00 t/s") != null); // 31 / 0.5
    try testing.expect(std.mem.indexOf(u8, done, "tg 39.90 t/s") != null); // 399 / 10
    try testing.expect(std.mem.indexOf(u8, done, "replayed") != null);

    // A turn that produced one token has no decode interval to divide by.
    var single: Status = .{};
    single.apply(.{ .turn_end = .{ .stop = .eos, .stats = .{ .generated = 1, .decode_seconds = 0.1 } } });
    try testing.expect(single.rates.decode == null);
}

test "a busy bar shows the loop's step against its budget" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var status: Status = .{ .busy = true, .message = "prefill", .step = 3, .budget = 16 };
    status.apply(.{ .status = .{ .phase = .prefill, .position = 10, .target = 100 } });
    const row = try painted(a, status, .{ .width = 160, .th = .{ .kind = .plain } });
    try testing.expect(std.mem.indexOf(u8, row, "step 3/16") != null);
    // An idle bar carries no step: the loop is not running.
    const idle: Status = .{};
    const idle_row = try painted(a, idle, .{ .width = 160, .th = .{ .kind = .plain } });
    try testing.expect(std.mem.indexOf(u8, idle_row, "step ") == null);
}

test "the bar truncates to its width instead of wrapping, in either glyph set" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const status: Status = .{ .message = "generating", .context_used = 30000, .context_capacity = 32768, .effort = "xhigh" };
    for ([_]theme.GlyphSet{ .unicode, .ascii }) |glyphs| {
        for ([_]usize{ 20, 40, 100, 400 }) |width| {
            const row = try status.paint(a, .{ .width = width, .th = .{ .kind = .truecolor, .glyph_set = glyphs } });
            try testing.expectEqual(width, view.styledWidth(row));
            try testing.expect(std.mem.indexOfScalar(u8, row, '\n') == null);
        }
    }
}
