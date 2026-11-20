//! The status bar: one row of measured state, painted as one string.
//!
//! Everything on it is a measurement or a setting, never an estimate dressed
//! up as a measurement — the rule `bench` set and the surface inherits. A rate
//! that was not measured prints `—`, and the prefill countdown is marked `~`
//! because it is extrapolated from the rate measured so far *this turn*, never
//! from a constant.
//!
//! The value is filled from the turn's `status` events (`apply`) plus the
//! settings the agent knows (context window, effort, the speculative switch,
//! the cache precision, the backend, the word on the left). `paint` is pure:
//! give it a width, a palette, a spinner frame, and the seconds elapsed, and
//! it returns the row. That is what lets the bar's layout be tested without a
//! model, a clock, or a terminal.
//!
//! Two groups: the measurements on the left, the settings on the right,
//! right-aligned. A bar too narrow for both drops settings cells from the
//! right end, then whole measurement cells from the right (the replay, `tg`,
//! `pp`, `in/out`), and truncates at the column only when the phase cell and
//! `ctx` alone do not fit; a wrapped bar would occupy two rows and break the
//! cursor arithmetic in `screen.zig`.
//! The row is padded to exactly `width` display cells with the escape-aware
//! width (`view.styledWidth`), so the painted background spans it.
const std = @import("std");
const theme = @import("theme.zig");
const view = @import("view.zig");
const event_mod = @import("event.zig");
const Event = event_mod.Event;

/// Padding for the bar's background; terminals clamp width to 400.
const spaces: [512]u8 = @splat(' ');

pub const Paint = struct {
    width: usize,
    th: theme.Theme,
    /// Spinner frame, advanced by the caller on every repaint.
    frame: usize = 0,
    /// Seconds since the turn started, at paint time. The caller has the
    /// clock; this module has none.
    elapsed_seconds: f64 = 0,
};

/// A running tool: its display name and the seconds since its call
/// started, from the caller's clock.
pub const Running = struct { name: []const u8, seconds: f64 };

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
    /// Messages typed during the turn and waiting for its next step.
    steering: usize = 0,
    /// Set when the turn re-prefilled the conversation.
    replay: ?event_mod.Replay = null,
    /// The tool call running now, if any: it takes the phase cell, since
    /// the last model phase says nothing while a command holds the turn.
    running: ?Running = null,
    /// Mean accepted drafts per verify batch, from the turn's engine timing;
    /// null on an ordinary run. Shown inside the `spec` cell while non-null.
    accepted_per_step: ?f64 = null,
    /// The speculative switch as the session runs it: on only when a draft
    /// source is loaded. `draft_length` is shown beside it when on.
    speculative: bool = false,
    draft_length: usize = 0,
    /// The cache precision and the executor, as their names.
    kv: []const u8 = "",
    backend: []const u8 = "",

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
                self.replay = end.stats.replay;
                self.accepted_per_step = end.stats.accepted_per_step;
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

    /// `text` without control characters, a newline as a space: one row.
    fn plain(a: std.mem.Allocator, text: []const u8) ![]const u8 {
        var out: std.Io.Writer.Allocating = .init(a);
        try view.safe(&out.writer, text);
        const cleaned = out.written();
        std.mem.replaceScalar(u8, cleaned, '\n', ' ');
        return cleaned;
    }

    pub fn paint(self: Status, a: std.mem.Allocator, options: Paint) ![]const u8 {
        const th = options.th;
        const gl = th.glyphs();
        // The measurements, each cell its own string: the first (the phase,
        // the step, steering) and `ctx` always stay; the rest drop whole from
        // the right once every setting has gone.
        var head: std.Io.Writer.Allocating = .init(a);
        const spin: []const u8 = if (self.busy) gl.spinner[options.frame % gl.spinner.len] else gl.idle;
        // The bar is a raw row: its free text (the model's name comes from
        // the file) is cleaned here.
        if (self.busy and self.running != null) {
            const tool = self.running.?;
            try head.writer.print(" {s} running {s} · {d:.0}s", .{ spin, try plain(a, tool.name), @floor(tool.seconds) });
        } else try head.writer.print(" {s} {s}", .{ spin, try plain(a, self.message) });
        if (self.busy and self.running != null) {} else if (self.busy and self.phase == .prefill and self.target > 0 and self.position < self.target) {
            try head.writer.print(" {s}{d}/{d}{s}", .{ th.paint(.progress), self.position, self.target, theme.fg_default });
            if (self.eta(options.elapsed_seconds)) |seconds_left| try head.writer.print(" ~{d:.0}s", .{seconds_left});
        } else if (self.busy and self.phase == .decode) {
            try head.writer.print(" {s}{d}{s} out, {d:.0}s", .{ th.paint(.progress), self.generated, theme.fg_default, options.elapsed_seconds });
        }
        if (self.busy and self.budget > 0)
            try head.writer.print(" {s} step {d}/{d}", .{ gl.table_bar, self.step, self.budget });
        if (self.busy and self.steering > 0)
            try head.writer.print(" {s} steering ×{d}", .{ gl.table_bar, self.steering });
        var measures: std.ArrayList([]const u8) = .empty;
        try measures.append(a, head.written());
        try measures.append(a, try a.print("{s} ctx {s}{d}/{d}{s}", .{ gl.context, th.paint(.accent), self.context_used, self.context_capacity, theme.fg_default }));
        try measures.append(a, try a.print("{s} in {d} out {d}", .{ gl.tokens, self.prompt_tokens, self.generated }));
        try measures.append(a, try rateCell(a, gl.prefill, "pp", self.rates.prefill));
        try measures.append(a, try rateCell(a, gl.decode, "tg", self.rates.decode));
        if (self.replay) |r| if (r.prefilled >= replay_shown) {
            var cell: std.Io.Writer.Allocating = .init(a);
            try cell.writer.writeAll("re-read ");
            try writeCount(&cell.writer, r.prefilled);
            try cell.writer.print(" tokens ({s})", .{causePhrase(r.cause)});
            try measures.append(a, cell.written());
        };

        // The settings, each cell its own string so the narrow bar can drop
        // them one at a time from the right, before any measurement.
        var cells: std.ArrayList([]const u8) = .empty;
        try cells.append(a, try a.print("{s} think {s}{s}{s}", .{ gl.effort, th.paint(.accent), self.effort, theme.fg_default }));
        if (self.speculative) {
            if (self.accepted_per_step) |accepted|
                try cells.append(a, try a.print("speculative {s}on{s} {d} · {d:.2}/step", .{ th.paint(.accent), theme.fg_default, self.draft_length, accepted }))
            else
                try cells.append(a, try a.print("speculative {s}on{s} {d}", .{ th.paint(.accent), theme.fg_default, self.draft_length }));
        } else try cells.append(a, "speculative off");
        if (self.kv.len != 0) try cells.append(a, try a.print("kv {s}", .{self.kv}));
        if (self.backend.len != 0) try cells.append(a, self.backend);
        if (self.model.len != 0) try cells.append(a, try plain(a, self.model));

        var left = try joinCells(a, measures.items, gl.table_bar);
        var left_width = view.styledWidth(left);
        var keep = cells.items.len;
        var right: []const u8 = "";
        while (keep > 0) : (keep -= 1) {
            right = try joinCells(a, cells.items[0..keep], gl.table_bar);
            // One cell of gap at least, and one of margin at the right edge.
            if (left_width + 1 + view.styledWidth(right) + 1 <= options.width) break;
        }
        if (keep == 0) {
            right = "";
            var kept = measures.items.len;
            while (left_width > options.width and kept > always_kept) {
                kept -= 1;
                left = try joinCells(a, measures.items[0..kept], gl.table_bar);
                left_width = view.styledWidth(left);
            }
        }
        const right_width = view.styledWidth(right);
        const body = if (right.len == 0) left else blk: {
            const gap = options.width - left_width - right_width - 1;
            break :blk try std.mem.concat(a, u8, &.{ left, spaces[0..@min(gap, spaces.len)], right, " " });
        };
        const wrapped = try view.wrapStyled(a, body, options.width, .character);
        const pad = options.width -| view.styledWidth(wrapped[0]);
        return std.mem.concat(a, u8, &.{ wrapped[0], spaces[0..@min(pad, spaces.len)] });
    }
};

/// The fewest re-read tokens the bar names: a smaller replay (a resume's,
/// a short cancelled step's) is noise on a row read every turn. Every
/// replay, whatever its size, stays in the session file and `--print --json`.
pub const replay_shown = 1024;

/// A replay's cause (`completer.Replay.Cause`, by name) in plain words; an
/// unknown name is shown as it is.
pub fn causePhrase(cause: []const u8) []const u8 {
    const phrases = [_][2][]const u8{
        .{ "resumed", "after a resume" },
        .{ "cancel", "after a cancel" },
        .{ "rewrite", "an earlier message changed" },
        .{ "rewind_out_of_reach", "after the window moved on" },
        .{ "effort", "the effort changed" },
        .{ "context", "the context size changed" },
        .{ "model", "the model changed" },
        .{ "failure", "after a failure" },
    };
    for (phrases) |pair| if (std.mem.eql(u8, pair[0], cause)) return pair[1];
    return cause;
}

/// A token count for a narrow row: exact under a thousand, `2.3k` above.
pub fn writeCount(w: *std.Io.Writer, n: usize) !void {
    if (n < 1000) return w.print("{d}", .{n});
    try w.print("{d:.1}k", .{@as(f64, @floatFromInt(n)) / 1000.0});
}

fn joinCells(a: std.mem.Allocator, cells: []const []const u8, bar: []const u8) ![]const u8 {
    var w: std.Io.Writer.Allocating = .init(a);
    for (cells, 0..) |cell, i| {
        if (i > 0) try w.writer.print(" {s} ", .{bar});
        try w.writer.writeAll(cell);
    }
    return w.written();
}

/// The measurement cells a narrow bar never drops: the phase cell and `ctx`.
const always_kept = 2;

/// One rate cell (`⇤ pp 84.26 t/s`), `—` for a rate not measured.
fn rateCell(a: std.mem.Allocator, glyph: []const u8, name: []const u8, value: ?f64) ![]const u8 {
    if (value) |v| return a.print("{s} {s} {d:.2} t/s", .{ glyph, name, v });
    return a.print("{s} {s} — t/s", .{ glyph, name });
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
    try testing.expect(std.mem.indexOf(u8, row, "speculative off") != null);
    try testing.expect(std.mem.indexOf(u8, row, "re-read") == null);
    // The row is exactly as wide as the bar, so its background spans it.
    try testing.expectEqual(@as(usize, 100), view.styledWidth(try status.paint(a, .{ .width = 100, .th = .{ .kind = .plain } })));
}

test "the settings sit at the right edge, every one of them on a wide bar" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const status: Status = .{ .context_used = 934, .context_capacity = 29696, .effort = "low", .speculative = true, .draft_length = 4, .kv = "f16", .backend = "metal", .model = "qwen3.8-27b" };
    const row = try painted(a, status, .{ .width = 200, .th = .{ .kind = .plain } });
    try testing.expectEqual(@as(usize, 200), view.width(row));
    try testing.expect(std.mem.endsWith(u8, row, "✦ think low │ speculative on 4 │ kv f16 │ metal │ qwen3.8-27b "));
    try testing.expect(std.mem.startsWith(u8, row, " ◆ ready │ ▤ ctx 934/29696 │ ⇅ in 0 out 0 │ ⇤ pp — t/s │ ⇥ tg — t/s"));
    // The gap between the groups is blank.
    const left_end = std.mem.lastIndexOf(u8, row, "t/s").? + 3;
    const right_start = std.mem.indexOf(u8, row, "✦").?;
    try testing.expect(std.mem.allEqual(u8, row[left_end..right_start], ' '));
    // The accepted rate joins the spec cell after a speculative turn.
    var measured = status;
    measured.accepted_per_step = 3.13;
    const after = try painted(a, measured, .{ .width = 200, .th = .{ .kind = .plain } });
    try testing.expect(std.mem.indexOf(u8, after, "speculative on 4 · 3.13/step") != null);
}

test "a narrow bar drops settings from the right, then truncates the measurements" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const status: Status = .{ .context_used = 934, .context_capacity = 29696, .effort = "low", .speculative = true, .draft_length = 4, .kv = "f16", .backend = "metal", .model = "qwen3.8-27b" };
    // The measurements are 67 cells; the settings need 1 cell of gap and 1
    // of margin: 120 holds up to the backend, 100 up to the switch, 80 the
    // effort alone.
    const at120 = try painted(a, status, .{ .width = 120, .th = .{ .kind = .plain } });
    try testing.expect(std.mem.indexOf(u8, at120, "kv f16 │ metal") != null);
    try testing.expect(std.mem.indexOf(u8, at120, "qwen3.8-27b") == null);
    const at100 = try painted(a, status, .{ .width = 100, .th = .{ .kind = .plain } });
    try testing.expect(std.mem.indexOf(u8, at100, "speculative on 4") != null);
    try testing.expect(std.mem.indexOf(u8, at100, "kv f16") == null);
    const at80 = try painted(a, status, .{ .width = 80, .th = .{ .kind = .plain } });
    try testing.expect(std.mem.indexOf(u8, at80, "think low") != null);
    try testing.expect(std.mem.indexOf(u8, at80, "speculative") == null);
    // Then whole measurement cells from the right: `tg` (13 cells) goes at
    // 60, `pp` and `in/out` at 40, and what stays is never cut.
    const at60 = try painted(a, status, .{ .width = 60, .th = .{ .kind = .plain } });
    try testing.expect(std.mem.indexOf(u8, at60, "think") == null);
    try testing.expect(std.mem.indexOf(u8, at60, "⇤ pp — t/s ") != null);
    try testing.expect(std.mem.indexOf(u8, at60, "tg") == null);
    const at40 = try painted(a, status, .{ .width = 40, .th = .{ .kind = .plain } });
    try testing.expectEqualStrings(" ◆ ready │ ▤ ctx 934/29696", std.mem.trimEnd(u8, at40, " "));
    // Only the phase cell and `ctx` are ever truncated.
    const at20 = try painted(a, status, .{ .width = 20, .th = .{ .kind = .plain } });
    try testing.expect(std.mem.startsWith(u8, at20, " ◆ ready │ ▤ ctx"));
    for ([_]usize{ 200, 100, 80, 60, 40, 20 }) |width| {
        try testing.expectEqual(width, view.styledWidth(try status.paint(a, .{ .width = width, .th = .{ .kind = .truecolor } })));
    }
}

test "a busy bar at 80 columns drops a rate whole rather than cut it mid-word" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var status: Status = .{ .busy = true, .message = "generating", .phase = .decode, .generated = 58, .step = 2, .budget = 25, .context_used = 2310, .context_capacity = 13312, .prompt_tokens = 2310, .effort = "low", .backend = "metal" };
    status.rates = .{ .prefill = 84.26, .decode = 20.25 };
    const at80 = try painted(a, status, .{ .width = 80, .th = .{ .kind = .plain }, .elapsed_seconds = 6 });
    try testing.expectEqual(@as(usize, 80), view.width(at80));
    try testing.expect(std.mem.endsWith(u8, std.mem.trimEnd(u8, at80, " "), "│ ⇅ in 2310 out 58"));
    try testing.expect(std.mem.indexOf(u8, at80, "pp") == null and std.mem.indexOf(u8, at80, "tg") == null);
    try testing.expect(std.mem.indexOf(u8, at80, "think") == null);
    const at60 = try painted(a, status, .{ .width = 60, .th = .{ .kind = .plain }, .elapsed_seconds = 6 });
    try testing.expectEqualStrings(" ⠋ generating 58 out, 6s │ step 2/25 │ ▤ ctx 2310/13312", std.mem.trimEnd(u8, at60, " "));
}

test "a replay keeps its cell over the settings, and drops before the rates" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const status: Status = .{ .context_used = 4900, .context_capacity = 32768, .effort = "low", .speculative = true, .draft_length = 4, .kv = "f16", .backend = "metal", .model = "qwen3.8-27b", .replay = .{ .cause = "rewrite", .prefilled = 4900 } };
    // At 160 the settings go first, from the right; the replay stays whole.
    const at160 = try painted(a, status, .{ .width = 160, .th = .{ .kind = .plain } });
    try testing.expect(std.mem.indexOf(u8, at160, "│ re-read 4.9k tokens (an earlier message changed)") != null);
    try testing.expect(std.mem.indexOf(u8, at160, "qwen3.8-27b") == null);
    try testing.expect(std.mem.endsWith(u8, std.mem.trimEnd(u8, at160, " "), "think low │ speculative on 4 │ kv f16"));
    // At 100 every setting has gone, then the replay, and the rates stay.
    const at100 = try painted(a, status, .{ .width = 100, .th = .{ .kind = .plain } });
    try testing.expect(std.mem.indexOf(u8, at100, "re-read") == null);
    try testing.expect(std.mem.indexOf(u8, at100, "think") == null);
    try testing.expect(std.mem.indexOf(u8, at100, "⇥ tg — t/s") != null);
}

test "a model name or message from a file cannot write escapes on the raw bar" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const status: Status = .{ .message = "bad\x1b]0;x\x07\nnews", .model = "evil\x1b[2J\u{9b}2Jname" };
    const row = try status.paint(a, .{ .width = 200, .th = .{ .kind = .plain } });
    try testing.expect(std.mem.indexOfScalar(u8, row, 0x07) == null);
    try testing.expect(std.mem.indexOf(u8, row, "\x1b]") == null);
    try testing.expect(std.mem.indexOf(u8, row, "\x1b[2J") == null);
    try testing.expect(std.mem.indexOf(u8, row, "\u{9b}") == null);
    try testing.expect(std.mem.indexOfScalar(u8, row, '\n') == null);
    try testing.expect(std.mem.indexOf(u8, row, "evil[2J2Jname") != null);
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
    var status: Status = .{ .busy = true, .message = "generating", .generated = 12, .speculative = true, .draft_length = 4 };
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
        .accepted_per_step = 2.33,
        .replay = .{ .cause = "cancel", .prefilled = 31, .restored = 15800 },
    } } });
    const done = try painted(a, status, .{ .width = 160, .th = .{ .kind = .plain } });
    try testing.expect(!status.busy);
    try testing.expect(std.mem.indexOf(u8, done, "in 31 out 400") != null);
    try testing.expect(std.mem.indexOf(u8, done, "pp 62.00 t/s") != null); // 31 / 0.5
    try testing.expect(std.mem.indexOf(u8, done, "tg 39.90 t/s") != null); // 399 / 10
    try testing.expect(std.mem.indexOf(u8, done, "speculative on 4 · 2.33/step") != null);
    // A 31-token replay is below the bar's threshold: its figures stay in
    // the session file, not on the row.
    try testing.expect(std.mem.indexOf(u8, done, "re-read") == null);
    try testing.expect(std.mem.indexOf(u8, done, "15800") == null);

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
    try testing.expect(std.mem.indexOf(u8, row, "steering") == null);
    status.steering = 2;
    const steering_row = try painted(a, status, .{ .width = 160, .th = .{ .kind = .plain } });
    try testing.expect(std.mem.indexOf(u8, steering_row, "steering ×2") != null);
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

test "a running tool takes the phase cell, timed from its call" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var status: Status = .{ .busy = true, .message = "generating", .phase = .decode, .generated = 58, .step = 2, .budget = 25, .context_used = 2310, .context_capacity = 13312, .prompt_tokens = 2310 };
    const generating = try painted(a, status, .{ .width = 160, .th = .{ .kind = .plain }, .elapsed_seconds = 6 });
    try testing.expect(std.mem.startsWith(u8, generating, " ⠋ generating 58 out, 6s │ step 2/25 │ ▤ ctx 2310/13312"));
    status.running = .{ .name = "Bash", .seconds = 6.8 };
    const running = try painted(a, status, .{ .width = 160, .th = .{ .kind = .plain }, .elapsed_seconds = 31 });
    try testing.expect(std.mem.startsWith(u8, running, " ⠋ running Bash · 6s │ step 2/25 │ ▤ ctx 2310/13312"));
    try testing.expect(std.mem.indexOf(u8, running, "out, ") == null);
    // An idle bar never names a tool.
    status.busy = false;
    try testing.expect(std.mem.indexOf(u8, try painted(a, status, .{ .width = 160, .th = .{ .kind = .plain } }), "running") == null);
}

test "the bar names a replay only from 1,024 re-read tokens, in plain words" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var status: Status = .{ .context_used = 2310, .context_capacity = 12288 };
    status.replay = .{ .cause = "cancel", .prefilled = 260, .restored = 1964 };
    const small = try painted(a, status, .{ .width = 160, .th = .{ .kind = .plain } });
    try testing.expect(std.mem.indexOf(u8, small, "re-read") == null);
    try testing.expect(std.mem.indexOf(u8, small, "1964") == null);
    status.replay = .{ .cause = "cancel", .prefilled = 2300, .restored = 1964 };
    const large = try painted(a, status, .{ .width = 160, .th = .{ .kind = .plain } });
    try testing.expect(std.mem.indexOf(u8, large, "│ re-read 2.3k tokens (after a cancel)") != null);
    try testing.expect(std.mem.indexOf(u8, large, "restored") == null);
    status.replay = .{ .cause = "rewrite", .prefilled = replay_shown };
    try testing.expect(std.mem.indexOf(u8, try painted(a, status, .{ .width = 160, .th = .{ .kind = .plain } }), "re-read 1.0k tokens (an earlier message changed)") != null);
    status.replay = .{ .cause = "effort", .prefilled = replay_shown - 1 };
    try testing.expect(std.mem.indexOf(u8, try painted(a, status, .{ .width = 160, .th = .{ .kind = .plain } }), "re-read") == null);
}
