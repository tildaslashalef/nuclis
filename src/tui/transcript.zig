//! The transcript: events in, rows out, and the rule that a completed block
//! is written to the terminal exactly once.
//!
//! A finished turn belongs to the terminal's scrollback and is never
//! repainted, while a bounded live region at the bottom holds what is still
//! moving (agent-spec § Rendering model). This module draws that line: it
//! keeps the current turn's blocks, each *closed* or open, and offers three
//! views:
//!
//! - `takeClosed` — rows for everything closed since the last call, marked
//!   written so it cannot return twice; an answer flushes block by block as
//!   its markdown blocks close, so it moves to the scrollback while written.
//! - `liveRows` — rows for what is still open, tail-clamped under a
//!   `… N lines above` marker; nothing is lost, it appears in full on close.
//! - `replayRows` — every written block again, for a fold toggle or a resize;
//!   the caller rewrites as many rows as the count changed by.
//!
//! The transcript copies what it keeps: events borrow their strings
//! (`event.zig`), blocks own theirs, and the three views allocate rows from a
//! short-lived caller arena. `Block` is a tagged union, so the renderer's
//! switch is exhaustive.
const std = @import("std");
const screen = @import("screen.zig");
const theme = @import("theme.zig");
const markdown = @import("markdown.zig");
const view = @import("view.zig");
const diff = @import("diff.zig");
const Event = @import("event.zig").Event;
const event_mod = @import("event.zig");

const Allocator = std.mem.Allocator;
pub const Row = screen.Row;

/// Above this many display rows a diff folds under an `… N lines` marker; a
/// whole-file rewrite should not push the live region off the screen.
pub const max_diff_rows: usize = 40;
/// At or above this terminal width a diff is drawn side by side (old | new);
/// below it there is not room for two readable panes, so the unified
/// single-column form is used instead.
pub const side_by_side_min_width: usize = 96;

/// One block of the transcript. The kinds are the specification's, and phase
/// 2 produces the three this phase only renders (tool call, tool result,
/// diff).
pub const Block = union(enum) {
    user: []u8,
    thinking: Thinking,
    answer: Answer,
    tool_call: struct { id: event_mod.Id, name: []u8, summary: []u8, detail: ?[]u8 = null, running: bool = true, failed: bool = false },
    tool_result: struct { id: event_mod.Id, text: []u8, truncated: bool, is_error: bool, summary: []u8 },
    diff: Diff,
    notice: []u8,
    /// A command's answer, at the terminal's own foreground.
    info: []u8,
    /// The end of a turn: its stop marker, if it needs one, and the blank row
    /// that separates it from the next turn.
    turn_end: struct { stop: event_mod.StopReason },
    /// What a run of tool calls amounted to (`Read 2 files, ran 1 shell
    /// command`), written where the model's text resumes after them.
    ops: []u8,

    pub const Thinking = struct {
        text: std.ArrayList(u8) = .empty,
        /// The step's reasoning time, from its start to the end of the
        /// channel; null while the block is open and after a cancelled turn
        /// (the bare label). Zero when a replayed session never kept it.
        seconds: ?f64 = null,
        closed: bool = false,
    };

    /// A mutation's change: the path and the structured rows, both owned.
    pub const Diff = struct { path: []u8, rows: []diff.Row };

    pub const Answer = struct {
        text: std.ArrayList(u8) = .empty,
        /// Bytes already written above the live region.
        flushed: usize = 0,
        closed: bool = false,
    };

    fn deinit(self: *Block, alloc: Allocator) void {
        switch (self.*) {
            .user, .notice, .info, .ops => |text| alloc.free(text),
            .thinking => |*t| t.text.deinit(alloc),
            .answer => |*a| a.text.deinit(alloc),
            .tool_call => |c| {
                alloc.free(c.name);
                alloc.free(c.summary);
                if (c.detail) |detail| alloc.free(detail);
            },
            .tool_result => |r| {
                alloc.free(r.text);
                alloc.free(r.summary);
            },
            .diff => |d| {
                alloc.free(d.path);
                diff.freeRows(alloc, d.rows);
            },
            .turn_end => {},
        }
    }

    /// Whether the block can still change. The two streaming kinds answer
    /// false until they close, and a tool call stays open until the result
    /// with the same id arrives, so its line keeps a spinner in the live
    /// region rather than being written to the scrollback before it settles.
    fn closed(self: Block) bool {
        return switch (self) {
            .thinking => |t| t.closed,
            .answer => |a| a.closed,
            .tool_call => |c| !c.running,
            else => true,
        };
    }
};

/// Tool calls counted by what they did, for the `ops` row.
pub const OpsCount = struct {
    reads: usize = 0,
    searches: usize = 0,
    commands: usize = 0,
    writes: usize = 0,

    fn any(self: OpsCount) bool {
        return self.reads + self.searches + self.commands + self.writes > 0;
    }

    fn add(self: *OpsCount, name: []const u8) void {
        if (std.mem.eql(u8, name, "read_file")) self.reads += 1 else if (std.mem.eql(u8, name, "grep") or std.mem.eql(u8, name, "glob")) self.searches += 1 else if (std.mem.eql(u8, name, "bash")) self.commands += 1 else if (isWrite(name)) self.writes += 1 else self.commands += 1;
    }

    /// `Read 2 files, searched 3 times, ran 1 shell command, wrote 1 file`.
    fn text(self: OpsCount, a: Allocator) ![]u8 {
        var w: std.Io.Writer.Allocating = .init(a);
        var parts: usize = 0;
        if (self.reads > 0) try part(&w.writer, &parts, "Read {d} file{s}", self.reads);
        if (self.searches > 0) try part(&w.writer, &parts, "searched {d} time{s}", self.searches);
        if (self.commands > 0) try part(&w.writer, &parts, "ran {d} shell command{s}", self.commands);
        if (self.writes > 0) try part(&w.writer, &parts, "wrote {d} file{s}", self.writes);
        const out = try w.toOwnedSlice();
        if (out.len > 0) out[0] = std.ascii.toUpper(out[0]);
        return out;
    }

    fn part(w: *std.Io.Writer, parts: *usize, comptime fmt: []const u8, n: usize) !void {
        if (parts.* > 0) try w.writeAll(", ");
        try w.print(fmt, .{ n, if (n == 1) "" else "s" });
        parts.* += 1;
    }
};

fn isWrite(name: []const u8) bool {
    return std.mem.eql(u8, name, "write_file") or std.mem.eql(u8, name, "edit_file");
}

/// What every rendering needs: the width to wrap at and the palette.
pub const Render = struct { width: usize, th: theme.Theme };

/// The live region's view: the same, plus the rows it can spend and the label
/// of a thinking block that has not closed yet. The label is the caller's
/// because it animates from a clock, and this module takes no `Io`.
pub const Live = struct {
    width: usize,
    th: theme.Theme,
    budget: usize,
    thinking_label: []const u8 = "",
    /// The spinner frame drawn before a running tool call. The caller owns it
    /// because it advances from a clock; the transcript has no `Io`. Empty
    /// falls back to the settled call glyph.
    spinner: []const u8 = "",
    /// The running dot's phase: lit or dim. The caller alternates it on its
    /// repaint cadence, so a long call blinks slowly.
    pulse: bool = true,
};

pub const Transcript = struct {
    alloc: Allocator,
    /// Tool calls since the last `ops` row, by kind, for the next one.
    ops: OpsCount = .{},
    blocks: std.ArrayList(Block) = .empty,
    /// Blocks fully written above the live region.
    written: usize = 0,
    /// Rows those blocks occupy, for a fold or resize replay.
    rows: usize = 0,
    /// Fold state of thinking blocks, toggled by Tab. It applies to whatever
    /// is rendered next; blocks already scrolled out of reach keep what they
    /// were printed with, which is the rendering model's rule.
    expanded: bool = false,

    pub fn deinit(self: *Transcript) void {
        self.clear();
        self.blocks.deinit(self.alloc);
    }

    /// Drops every block: a new turn, a new session. The rows already written
    /// stay in the terminal's scrollback, where they belong.
    pub fn reset(self: *Transcript) void {
        self.clear();
        self.written = 0;
        self.rows = 0;
        self.ops = .{};
    }

    fn clear(self: *Transcript) void {
        for (self.blocks.items) |*block| block.deinit(self.alloc);
        self.blocks.clearRetainingCapacity();
    }

    /// Closes the streaming blocks without ending the turn. The live path
    /// closes them with `turn_end`; a replay of a stored conversation emits
    /// each assistant step in turn and needs the next step's answer to open a
    /// fresh block rather than grow one already written.
    pub fn closeOpen(self: *Transcript) void {
        for (self.blocks.items) |*block| switch (block.*) {
            .thinking => |*t| t.closed = true,
            .answer => |*a| a.closed = true,
            else => {},
        };
    }

    /// Is anything still open, or unwritten?
    pub fn pending(self: *const Transcript) bool {
        return self.written < self.blocks.items.len;
    }

    // ----- events -----

    pub fn apply(self: *Transcript, e: Event) !void {
        switch (e) {
            .user => |text| try self.blocks.append(self.alloc, .{ .user = try self.alloc.dupe(u8, text) }),
            .notice => |text| try self.blocks.append(self.alloc, .{ .notice = try self.alloc.dupe(u8, text) }),
            .info => |text| try self.blocks.append(self.alloc, .{ .info = try self.alloc.dupe(u8, text) }),
            .thinking_delta => |text| {
                try self.summarizeOps();
                const block = try self.openThinking();
                try block.text.appendSlice(self.alloc, text);
            },
            .thinking_end => |seconds| {
                const block = try self.openThinking();
                block.seconds = seconds;
                block.closed = true;
            },
            .answer_delta => |text| {
                try self.summarizeOps();
                const block = try self.openAnswer();
                try block.text.appendSlice(self.alloc, text);
            },
            .tool_call => |call| {
                // A call ends the step's text: an answer streamed before it
                // would otherwise stay open until the turn ends and hold
                // everything after it in the live region.
                self.closeOpen();
                const name = try self.alloc.dupe(u8, call.name);
                errdefer self.alloc.free(name);
                const summary = try self.alloc.dupe(u8, call.summary);
                errdefer self.alloc.free(summary);
                const detail: ?[]u8 = if (call.detail) |detail| try self.alloc.dupe(u8, detail) else null;
                errdefer if (detail) |d| self.alloc.free(d);
                try self.blocks.append(self.alloc, .{ .tool_call = .{ .id = call.id, .name = name, .summary = summary, .detail = detail } });
                self.ops.add(call.name);
            },
            .tool_result => |result| {
                // The call this result answers is settled now: its line stops
                // animating and can be written to the scrollback.
                for (self.blocks.items) |*block| {
                    if (block.* == .tool_call and block.tool_call.id == result.id) {
                        block.tool_call.running = false;
                        block.tool_call.failed = result.is_error;
                    }
                }
                const text = try self.alloc.dupe(u8, result.text);
                errdefer self.alloc.free(text);
                const summary = try self.alloc.dupe(u8, result.summary);
                errdefer self.alloc.free(summary);
                try self.blocks.append(self.alloc, .{ .tool_result = .{ .id = result.id, .text = text, .truncated = result.truncated, .is_error = result.is_error, .summary = summary } });
            },
            .diff => |d| {
                const path = try self.alloc.dupe(u8, d.path);
                errdefer self.alloc.free(path);
                const rows = try diff.cloneRows(self.alloc, d.rows);
                errdefer diff.freeRows(self.alloc, rows);
                try self.blocks.append(self.alloc, .{ .diff = .{ .path = path, .rows = rows } });
            },
            // A status beat is the bar's, not the transcript's: it describes
            // the turn's progress, never a row of the conversation.
            .status => {},
            .turn_end => |end| {
                // Everything still open is final now, whatever it was waiting
                // for: a cancelled turn keeps the text it reached.
                for (self.blocks.items) |*block| switch (block.*) {
                    .thinking => |*t| t.closed = true,
                    .answer => |*a| a.closed = true,
                    else => {},
                };
                try self.summarizeOps();
                try self.blocks.append(self.alloc, .{ .turn_end = .{ .stop = end.stop } });
            },
        }
    }

    /// Writes the `ops` row for the calls made since the last one, if any.
    /// Called where the model's text resumes and at the end of a turn.
    fn summarizeOps(self: *Transcript) !void {
        if (!self.ops.any()) return;
        const text = try self.ops.text(self.alloc);
        errdefer self.alloc.free(text);
        try self.blocks.append(self.alloc, .{ .ops = text });
        self.ops = .{};
    }

    /// The open thinking block, created when there is none. Only the *last*
    /// block can be open, which is what keeps the rule intact: a block that
    /// has been written never changes again, so a model that reopened its
    /// reasoning channel after answering would start a second block rather
    /// than grow one whose rows are already in the scrollback.
    fn openThinking(self: *Transcript) !*Block.Thinking {
        if (self.blocks.items.len > 0) {
            const last = &self.blocks.items[self.blocks.items.len - 1];
            if (last.* == .thinking and !last.thinking.closed) return &last.thinking;
        }
        try self.blocks.append(self.alloc, .{ .thinking = .{} });
        return &self.blocks.items[self.blocks.items.len - 1].thinking;
    }

    fn openAnswer(self: *Transcript) !*Block.Answer {
        if (self.blocks.items.len > 0) {
            const last = &self.blocks.items[self.blocks.items.len - 1];
            if (last.* == .answer and !last.answer.closed) return &last.answer;
        }
        try self.blocks.append(self.alloc, .{ .answer = .{} });
        return &self.blocks.items[self.blocks.items.len - 1].answer;
    }

    // ----- views -----

    /// Rows for everything that closed since the last call, marked written.
    /// The caller inserts them above the live region; calling it twice
    /// returns the second time only what closed in between, which is the
    /// "exactly once" rule of the rendering model.
    pub fn takeClosed(self: *Transcript, a: Allocator, options: Render) ![]const Row {
        var out: std.ArrayList(Row) = .empty;
        while (self.written < self.blocks.items.len) {
            const block = &self.blocks.items[self.written];
            if (!block.closed()) {
                // An open answer still flushes the markdown blocks that have
                // closed inside it; everything after it waits its turn.
                if (block.* == .answer) try self.flushAnswer(a, &out, &block.answer, options);
                break;
            }
            try self.render(a, &out, block.*, options, .remainder);
            if (block.* == .answer) block.answer.flushed = block.answer.text.items.len;
            self.written += 1;
        }
        self.rows += out.items.len;
        return out.items;
    }

    /// The closed markdown blocks of an answer that is still being written.
    fn flushAnswer(self: *Transcript, a: Allocator, out: *std.ArrayList(Row), answer: *Block.Answer, options: Render) !void {
        const rest = answer.text.items[answer.flushed..];
        const blocks = markdown.split(rest);
        if (blocks.closed.len == 0) return;
        try self.pushMarkdown(a, out, blocks.closed, options);
        answer.flushed += blocks.closed.len;
    }

    /// Rows for what is still open, tail-clamped to the budget.
    pub fn liveRows(self: *Transcript, a: Allocator, options: Live) ![]const Row {
        var out: std.ArrayList(Row) = .empty;
        var budget = options.budget;
        const shape: Render = .{ .width = options.width, .th = options.th };
        for (self.blocks.items[self.written..]) |block| {
            var produced: std.ArrayList(Row) = .empty;
            switch (block) {
                .thinking => |t| {
                    // The open label animates, so it is the caller's; a block
                    // that closed but is not written yet wears its fold label.
                    // The text under it is shown only when unfolded, and only its tail.
                    const label = if (t.closed) try foldLabel(a, t, self.expanded, options.th) else options.thinking_label;
                    try produced.append(a, .{ .text = label, .style = .thinking_header });
                    if (self.expanded and t.text.items.len > 0) {
                        try produced.append(a, .{ .text = "" });
                        const text = try wrapped(a, t.text.items, options.width, .thinking);
                        try appendTail(a, &produced, text, @max(budget / 2, 1), options.th);
                    }
                },
                .answer => |ans| {
                    // Everything up to `flushed` is already in the scrollback:
                    // the region holds the block still being written, raw
                    // until it closes and can be styled.
                    try pushRaw(a, &produced, ans.text.items[ans.flushed..], options.width);
                },
                .tool_call => |call| {
                    // A running call keeps its dot pulsing here, in the
                    // region, until the result settles it into the scrollback.
                    if (call.running) {
                        try produced.append(a, try callRow(a, call, options.th, options.pulse));
                        if (call.detail) |detail| try pushDetail(a, &produced, options.th.glyphs(), detail, options.width, .tool_result);
                    } else try self.render(a, &produced, block, shape, .remainder);
                },
                else => try self.render(a, &produced, block, shape, .remainder),
            }
            const before = out.items.len;
            try appendTail(a, &out, produced.items, @max(budget, 1), options.th);
            budget -|= out.items.len - before;
        }
        return out.items;
    }

    /// Every written block rendered again: what a fold toggle or a resize
    /// rewrites. `rows` is updated to the new count, so the caller passes the
    /// old one to `Screen.replaceAbove` and this becomes the new height.
    pub fn replayRows(self: *Transcript, a: Allocator, options: Render) ![]const Row {
        var out: std.ArrayList(Row) = .empty;
        for (self.blocks.items[0..self.written]) |block| try self.render(a, &out, block, options, .written);
        // A partially flushed answer is written up to `flushed` and no further.
        if (self.written < self.blocks.items.len) {
            const block = self.blocks.items[self.written];
            if (block == .answer and block.answer.flushed > 0) {
                try self.pushMarkdown(a, &out, block.answer.text.items[0..block.answer.flushed], options);
            }
        }
        self.rows = out.items.len;
        return out.items;
    }

    // ----- block rendering -----

    /// Which part of a streaming block a render covers: what has *not* been
    /// written yet (an insertion above the region) or what *has* (a replay of
    /// the scrollback). Every other block kind renders the same either way.
    const Scope = enum { remainder, written };

    /// One block's rows. `render` is the definition of what each event kind
    /// looks like; the three views above only decide how much of it is shown
    /// and where it goes.
    fn render(self: *const Transcript, a: Allocator, out: *std.ArrayList(Row), block: Block, options: Render, scope: Scope) !void {
        const th = options.th;
        const gl = th.glyphs();
        switch (block) {
            .user => |text| {
                try pushWrapped(a, out, text, options.width, .user);
                try out.append(a, .{ .text = "" });
            },
            .thinking => |t| {
                try out.append(a, .{ .text = try foldLabel(a, t, self.expanded, th), .style = .thinking_header });
                if (self.expanded and t.text.items.len > 0) {
                    try out.append(a, .{ .text = "" });
                    try pushWrapped(a, out, t.text.items, options.width, .thinking);
                }
            },
            .answer => |ans| try self.pushMarkdown(a, out, switch (scope) {
                .remainder => ans.text.items[ans.flushed..],
                .written => ans.text.items[0..ans.flushed],
            }, options),
            .tool_call => |call| {
                try out.append(a, try callRow(a, call, th, true));
                if (call.detail) |detail| try pushDetail(a, out, gl, detail, options.width, .tool_result);
            },
            .ops => |text| try pushWrapped(a, out, text, options.width, .dim),
            .tool_result => |result| {
                // The result text is the model's; the reader gets one row
                // (the tool's summary), or the message when the call failed.
                if (result.is_error) {
                    var lines = std.mem.splitScalar(u8, std.mem.trimEnd(u8, result.text, "\n"), '\n');
                    var shown: usize = 0;
                    while (lines.next()) |line| : (shown += 1) {
                        if (shown == max_error_rows) {
                            try out.append(a, .{ .text = try std.fmt.allocPrint(a, "  {s}", .{gl.ellipsis}), .style = .dim });
                            break;
                        }
                        if (shown == 0) try pushDetail(a, out, gl, line, options.width, .error_text) else try pushWrapped(a, out, try std.fmt.allocPrint(a, "  {s}", .{line}), options.width, .error_text);
                    }
                } else if (result.summary.len > 0) {
                    try pushDetail(a, out, gl, result.summary, options.width, .tool_result);
                } else if (result.truncated) {
                    try pushDetail(a, out, gl, "truncated", options.width, .dim);
                }
            },
            .diff => |d| try self.renderDiff(a, out, d, options),
            .notice => |text| try pushWrapped(a, out, text, options.width, .dim),
            .info => |text| {
                try pushWrapped(a, out, text, options.width, null);
                try out.append(a, .{ .text = "" });
            },
            .turn_end => |end| {
                switch (end.stop) {
                    .token_budget => try out.append(a, .{ .text = "  — output budget reached (max-tokens)", .style = .dim }),
                    .cancelled => try out.append(a, .{ .text = "  — cancelled", .style = .dim }),
                    .context_limit => try out.append(a, .{ .text = "  — context full", .style = .dim }),
                    // A failure always arrives with a notice naming it, so
                    // the stop reason adds a row only where it says something
                    // the conversation does not.
                    .failure, .eos => {},
                }
                try out.append(a, .{ .text = "" });
            },
        }
    }

    /// A diff block: the path header, then the rows laid out for the width.
    /// Wide terminals get the two-pane view, narrow ones the unified form;
    /// a diff taller than `max_diff_rows` folds under an `… N lines` marker.
    fn renderDiff(self: *const Transcript, a: Allocator, out: *std.ArrayList(Row), d: Block.Diff, options: Render) !void {
        _ = self;
        const gl = options.th.glyphs();
        try out.append(a, .{ .text = try std.fmt.allocPrint(a, "{s} {s}", .{ gl.fold_open, d.path }), .style = .diff_header });
        var produced: std.ArrayList(Row) = .empty;
        if (options.width >= side_by_side_min_width) {
            try renderSideBySide(a, &produced, d.rows, options);
        } else {
            try renderUnified(a, &produced, d.rows, options.th);
        }
        if (produced.items.len > max_diff_rows and max_diff_rows > 1) {
            const hidden = produced.items.len - (max_diff_rows - 1);
            try out.appendSlice(a, produced.items[0 .. max_diff_rows - 1]);
            try out.append(a, .{ .text = try std.fmt.allocPrint(a, "{s} {d} lines", .{ gl.ellipsis, hidden }), .style = .dim });
        } else try out.appendSlice(a, produced.items);
    }

    fn pushMarkdown(self: *const Transcript, a: Allocator, out: *std.ArrayList(Row), text: []const u8, options: Render) !void {
        _ = self;
        const rendered = try markdown.render(a, text, options.width, options.th);
        for (rendered) |line| try out.append(a, .{ .text = line, .raw = true });
    }
};

/// The unified single-column form: `-`/`+`/space before each row, coloured.
fn renderUnified(a: Allocator, out: *std.ArrayList(Row), rows: []const diff.Row, th: theme.Theme) !void {
    for (rows) |row| {
        const sign: u8 = switch (row.kind) {
            .context => ' ',
            .remove => '-',
            .add => '+',
        };
        const base: theme.Style = switch (row.kind) {
            .context => .dim,
            .remove => .diff_remove,
            .add => .diff_add,
        };
        var w: std.Io.Writer.Allocating = .init(a);
        try w.writer.writeAll(th.paint(base));
        try w.writer.writeByte(sign);
        try writeHighlighted(&w.writer, row.text, row.change, base, th);
        try w.writer.writeAll(theme.reset);
        try out.append(a, .{ .text = w.written(), .raw = true });
    }
}

/// The two-pane form: old line number and text on the left, new on the right,
/// a removal paired with the addition it became on one display row.
fn renderSideBySide(a: Allocator, out: *std.ArrayList(Row), rows: []const diff.Row, options: Render) !void {
    var old_digits: usize = 1;
    var new_digits: usize = 1;
    for (rows) |row| {
        if (row.old_line) |n| old_digits = @max(old_digits, decimals(n));
        if (row.new_line) |n| new_digits = @max(new_digits, decimals(n));
    }
    const overhead = old_digits + new_digits + 1 + 3 + 1; // two numbers, spaces, " │ "
    const pane = if (options.width > overhead + 8) (options.width - overhead) / 2 else 8;

    var i: usize = 0;
    while (i < rows.len) {
        if (rows[i].kind == .context) {
            try sideRow(a, out, rows[i], rows[i], old_digits, new_digits, pane, options.th);
            i += 1;
            continue;
        }
        var removes_end = i;
        while (removes_end < rows.len and rows[removes_end].kind == .remove) removes_end += 1;
        var adds_end = removes_end;
        while (adds_end < rows.len and rows[adds_end].kind == .add) adds_end += 1;
        const removes = removes_end - i;
        const adds = adds_end - removes_end;
        const pairs = @max(removes, adds);
        for (0..pairs) |k| {
            const left: ?diff.Row = if (k < removes) rows[i + k] else null;
            const right: ?diff.Row = if (k < adds) rows[removes_end + k] else null;
            try sideRow(a, out, left, right, old_digits, new_digits, pane, options.th);
        }
        i = adds_end;
    }
}

fn sideRow(a: Allocator, out: *std.ArrayList(Row), left: ?diff.Row, right: ?diff.Row, old_digits: usize, new_digits: usize, pane: usize, th: theme.Theme) !void {
    var w: std.Io.Writer.Allocating = .init(a);
    try w.writer.writeAll(th.paint(.dim));
    try writeNumber(&w.writer, if (left) |row| row.old_line else null, old_digits);
    try w.writer.writeByte(' ');
    if (left) |row| try writePane(&w.writer, a, row, pane, th) else try writeBlankPane(&w.writer, pane);
    try w.writer.writeAll(th.paint(.dim));
    try w.writer.writeByte(' ');
    try w.writer.writeAll(th.glyphs().table_bar);
    try w.writer.writeByte(' ');
    try w.writer.writeAll(th.paint(.dim));
    try writeNumber(&w.writer, if (right) |row| row.new_line else null, new_digits);
    try w.writer.writeByte(' ');
    if (right) |row| try writePane(&w.writer, a, row, pane, th) else try writeBlankPane(&w.writer, pane);
    try w.writer.writeAll(theme.reset);
    try out.append(a, .{ .text = w.written(), .raw = true });
}

/// One pane's text: fitted to the pane width, the changed span reversed, then
/// padded with spaces so the separator column lines up.
fn writePane(w: *std.Io.Writer, a: Allocator, row: diff.Row, cells: usize, th: theme.Theme) !void {
    const fitted = try view.fit(a, row.text, cells);
    const base: theme.Style = switch (row.kind) {
        .add => .diff_add,
        .remove => .diff_remove,
        .context => .dim,
    };
    try w.writeAll(th.paint(base));
    try writeHighlighted(w, fitted, clampSpan(row.change, fitted.len), base, th);
    try w.writeAll(theme.reset);
    var pad = cells -| view.width(fitted);
    while (pad > 0) : (pad -= 1) try w.writeByte(' ');
}

fn writeBlankPane(w: *std.Io.Writer, cells: usize) !void {
    var pad = cells;
    while (pad > 0) : (pad -= 1) try w.writeByte(' ');
}

fn writeNumber(w: *std.Io.Writer, number: ?usize, width: usize) !void {
    var buffer: [24]u8 = undefined;
    const text = if (number) |n| std.fmt.bufPrint(&buffer, "{d}", .{n}) catch "?" else "";
    var pad = width -| text.len;
    while (pad > 0) : (pad -= 1) try w.writeByte(' ');
    try w.writeAll(text);
}

/// Writes `text`, wrapping the changed bytes in the highlight style and
/// returning to `base` afterwards. The bytes come from a file, so they are
/// sanitized here: the row is `raw` and the screen will not do it.
fn writeHighlighted(w: *std.Io.Writer, text: []const u8, change: ?diff.Span, base: theme.Style, th: theme.Theme) !void {
    const span = change orelse return view.safe(w, text);
    const start = @min(span.start, text.len);
    const end = @min(span.start + span.len, text.len);
    if (end <= start) return view.safe(w, text);
    try view.safe(w, text[0..start]);
    try w.writeAll(th.paint(.diff_change));
    try view.safe(w, text[start..end]);
    try w.writeAll(th.paint(base));
    try view.safe(w, text[end..]);
}

fn clampSpan(change: ?diff.Span, len: usize) ?diff.Span {
    const span = change orelse return null;
    const start = @min(span.start, len);
    const end = @min(span.start + span.len, len);
    if (end <= start) return null;
    return .{ .start = start, .len = end - start };
}

fn decimals(n: usize) usize {
    var value = n;
    var count: usize = 1;
    while (value >= 10) : (value /= 10) count += 1;
    return count;
}

/// The fold label of a closed thinking block, with its measured time when the
/// turn reached an answer.
fn foldLabel(a: Allocator, t: Block.Thinking, expanded: bool, th: theme.Theme) ![]const u8 {
    const gl = th.glyphs();
    const arrow = if (expanded) gl.fold_open else gl.fold_closed;
    const hint = if (expanded) "Tab to fold" else "Tab to unfold";
    if (t.seconds) |seconds| {
        if (seconds > 0) return std.fmt.allocPrint(a, "{s} Thought for {d:.1}s ({s})", .{ arrow, seconds, hint });
        return std.fmt.allocPrint(a, "{s} Thought ({s})", .{ arrow, hint });
    }
    return std.fmt.allocPrint(a, "{s} thinking ({s})", .{ arrow, hint });
}

/// The rows of an error result are bounded; the message's head is what the
/// reader needs, the rest is in the session file.
const max_error_rows: usize = 3;

/// One detail row under a tool call: the corner glyph and the text.
/// The call's row: the dot in the state's colour, then `Name(argument)`.
/// Raw, since the dot and the name carry different styles; the summary is
/// the model's argument and is sanitized here.
fn callRow(a: Allocator, call: anytype, th: theme.Theme, pulse: bool) !Row {
    const dot: theme.Style = if (call.running)
        (if (pulse) .op_running else .dim)
    else if (call.failed)
        .op_error
    else if (isWrite(call.name))
        .op_write
    else
        .op_ok;
    var w: std.Io.Writer.Allocating = .init(a);
    try w.writer.print("{s}{s}{s} {s}", .{ th.paint(dot), th.glyphs().dot, theme.reset, th.paint(.tool_call) });
    try view.safe(&w.writer, call.summary);
    try w.writer.writeAll(theme.reset);
    return .{ .text = w.written(), .raw = true };
}

fn pushDetail(a: Allocator, out: *std.ArrayList(Row), gl: theme.Glyphs, text: []const u8, width: usize, style: ?theme.Style) !void {
    try pushWrapped(a, out, try std.fmt.allocPrint(a, "{s} {s}", .{ gl.detail, text }), width, style);
}

fn pushWrapped(a: Allocator, out: *std.ArrayList(Row), text: []const u8, width: usize, style: ?theme.Style) !void {
    for (try view.lines(a, text, width, .word)) |line| try out.append(a, .{ .text = line, .style = style });
}

fn pushRaw(a: Allocator, out: *std.ArrayList(Row), text: []const u8, width: usize) !void {
    try pushWrapped(a, out, text, width, null);
}

fn wrapped(a: Allocator, text: []const u8, width: usize, style: ?theme.Style) ![]const Row {
    var out: std.ArrayList(Row) = .empty;
    try pushWrapped(a, &out, text, width, style);
    return out.items;
}

/// Appends at most the last `max` rows of `produced`, under a marker when the
/// head was dropped. What is hidden here is never lost: it appears in full
/// once its block closes and moves into the scrollback.
fn appendTail(a: Allocator, out: *std.ArrayList(Row), produced: []const Row, max: usize, th: theme.Theme) !void {
    if (produced.len > max and max > 1) {
        const hidden = produced.len - (max - 1);
        try out.append(a, .{ .text = try std.fmt.allocPrint(a, "{s} {d} lines above", .{ th.glyphs().ellipsis, hidden }), .style = .dim });
        try out.appendSlice(a, produced[hidden..]);
        return;
    }
    try out.appendSlice(a, produced[produced.len -| max..]);
}

// ----- tests -----

const testing = std.testing;

fn transcript() Transcript {
    return .{ .alloc = testing.allocator };
}

/// The text of every row, joined and stripped of escapes: the tests assert
/// on content and order, and the styles are pinned in `theme.zig`.
fn texts(a: Allocator, rows: []const Row) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    for (rows) |row| {
        try out.appendSlice(a, try stripped(a, row.text));
        try out.append(a, '\n');
    }
    return out.toOwnedSlice(a);
}

fn stripped(a: Allocator, text: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        if (text[i] == 0x1b) {
            i += 1;
            while (i < text.len and text[i] != 'm') i += 1;
            continue;
        }
        try out.append(a, text[i]);
    }
    return out.items;
}

test "a closed block is written exactly once" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tr = transcript();
    defer tr.deinit();
    const options: Render = .{ .width = 40, .th = .{ .kind = .plain } };

    try tr.apply(.{ .user = "hello there" });
    const first = try tr.takeClosed(a, options);
    try testing.expectEqual(@as(usize, 2), first.len); // the prompt and its blank row
    try testing.expectEqualStrings("hello there", first[0].text);
    try testing.expectEqual(@as(usize, 2), tr.rows);
    // Nothing new: the second call returns nothing at all.
    try testing.expectEqual(@as(usize, 0), (try tr.takeClosed(a, options)).len);
    try testing.expectEqual(@as(usize, 2), tr.rows);
}

test "an answer flushes block by block while it streams" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tr = transcript();
    defer tr.deinit();
    const options: Render = .{ .width = 40, .th = .{ .kind = .plain } };

    try tr.apply(.{ .answer_delta = "First paragraph." });
    // Nothing has closed yet: the paragraph is still being written.
    try testing.expectEqual(@as(usize, 0), (try tr.takeClosed(a, options)).len);
    const live = try tr.liveRows(a, .{ .width = 40, .th = options.th, .budget = 10 });
    try testing.expectEqualStrings("First paragraph.", live[0].text);

    // A blank line closes it, and only it.
    try tr.apply(.{ .answer_delta = "\n\nSecond" });
    const flushed = try tr.takeClosed(a, options);
    const written = try texts(a, flushed);
    try testing.expectEqualStrings("First paragraph.\n\n", written);
    // The open block is what the live region shows now.
    const rest = try tr.liveRows(a, .{ .width = 40, .th = options.th, .budget = 10 });
    try testing.expectEqualStrings("Second\n", try texts(a, rest));

    // The turn ends: the rest of the answer and the closing blank row follow,
    // and nothing that was already written repeats.
    try tr.apply(.{ .turn_end = .{ .stop = .eos } });
    try testing.expectEqualStrings("Second\n\n", try texts(a, try tr.takeClosed(a, options)));
    try testing.expectEqual(@as(usize, 0), (try tr.liveRows(a, .{ .width = 40, .th = options.th, .budget = 10 })).len);
}

test "the replay renders exactly what was written, with the current fold state" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tr = transcript();
    defer tr.deinit();
    const options: Render = .{ .width = 40, .th = .{ .kind = .plain } };

    try tr.apply(.{ .user = "why?" });
    try tr.apply(.{ .thinking_delta = "because" });
    try tr.apply(.{ .thinking_end = 3.42 });
    try tr.apply(.{ .answer_delta = "Because.\n\n" });
    _ = try tr.takeClosed(a, options);
    const folded = try texts(a, try tr.replayRows(a, options));
    try testing.expect(std.mem.indexOf(u8, folded, "▸ Thought for 3.4s (Tab to unfold)") != null);
    try testing.expect(std.mem.indexOf(u8, folded, "because") == null);
    const rows_folded = tr.rows;

    // Unfolding shows the reasoning text and makes the block taller; the row
    // count follows, which is what the screen needs to rewrite it.
    tr.expanded = true;
    const open = try texts(a, try tr.replayRows(a, options));
    try testing.expect(std.mem.indexOf(u8, open, "▾ Thought for 3.4s (Tab to fold)") != null);
    try testing.expect(std.mem.indexOf(u8, open, "because") != null);
    try testing.expect(tr.rows > rows_folded);
}

test "a flushed answer replays as one render of the same bytes" {
    // The scrollback holds the concatenation of the flushed segments; a fold
    // toggle rewrites it as one render. The two must produce the same rows,
    // or the replay would drift from what is on the screen.
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tr = transcript();
    defer tr.deinit();
    const options: Render = .{ .width = 30, .th = .{ .kind = .plain } };
    const document = "# Heading\n\nA paragraph that is long enough to wrap twice over.\n\n- one\n- two\n\n```zig\nconst x = 1;\n```\n\nDone.\n";
    var streamed: std.ArrayList(Row) = .empty;
    for (document) |byte| {
        try tr.apply(.{ .answer_delta = &.{byte} });
        try streamed.appendSlice(a, try tr.takeClosed(a, options));
    }
    try tr.apply(.{ .turn_end = .{ .stop = .eos } });
    try streamed.appendSlice(a, try tr.takeClosed(a, options));
    const replayed = try tr.replayRows(a, options);
    // The replay leaves out the turn's closing rows, which belong to the
    // block after the answer; compare what the answer produced.
    const one = try texts(a, replayed);
    const many = try texts(a, streamed.items);
    try testing.expect(std.mem.startsWith(u8, many, one));
    // The code block is there, in the fenced block's own styling (the
    // highlighter breaks the literal text up, so the plain part is enough).
    try testing.expect(std.mem.indexOf(u8, one, "x = ") != null);
}

test "every block kind renders, including the ones phase 2 produces" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tr = transcript();
    defer tr.deinit();
    const options: Render = .{ .width = 60, .th = .{ .kind = .plain } };
    const changed = [_]diff.Row{
        .{ .old_line = 1, .new_line = 1, .kind = .context, .text = "context" },
        .{ .old_line = 2, .kind = .remove, .text = "old" },
        .{ .new_line = 2, .kind = .add, .text = "new" },
    };
    try tr.apply(.{ .user = "edit it" });
    try tr.apply(.{ .tool_call = .{ .id = 7, .name = "read_file", .summary = "Reading src/main.zig" } });
    try tr.apply(.{ .tool_result = .{ .id = 7, .text = "line one\nline two\n", .truncated = true, .is_error = false, .summary = "lines 1 to 2 of 9 · truncated, continue with offset=3" } });
    try tr.apply(.{ .diff = .{ .path = "src/main.zig", .rows = &changed } });
    try tr.apply(.{ .tool_result = .{ .id = 8, .text = "no such file", .truncated = false, .is_error = true } });
    try tr.apply(.{ .notice = "older turns dropped from context to fit" });
    try tr.apply(.{ .info = "  /help  keys and commands" });
    try tr.apply(.{ .status = .{ .phase = .decode } }); // never a row
    try tr.apply(.{ .turn_end = .{ .stop = .token_budget } });
    const rows = try tr.takeClosed(a, options);
    const s = try texts(a, rows);
    try testing.expect(std.mem.indexOf(u8, s, "● Reading src/main.zig") != null); // the summary as given; describe() shapes the real one
    // The result's text is the model's; the reader sees the tool's one row.
    try testing.expect(std.mem.indexOf(u8, s, "line one") == null);
    try testing.expect(std.mem.indexOf(u8, s, "└ lines 1 to 2 of 9 · truncated, continue with offset=3") != null);
    try testing.expect(std.mem.indexOf(u8, s, "▾ src/main.zig") != null);
    try testing.expect(std.mem.indexOf(u8, s, "-old") != null);
    try testing.expect(std.mem.indexOf(u8, s, "+new") != null);
    try testing.expect(std.mem.indexOf(u8, s, "└ no such file") != null);
    try testing.expect(std.mem.indexOf(u8, s, "older turns dropped") != null);
    // A command's answer is content: it carries no style at all, so it is
    // painted at the terminal's own foreground rather than dimmed.
    for (rows) |row| {
        if (std.mem.indexOf(u8, row.text, "keys and commands") != null) try testing.expect(row.style == null);
    }
    try testing.expect(std.mem.indexOf(u8, s, "output budget reached") != null);
    // The styles a reader needs to tell them apart, in the order rendered.
    // The call row's and the diff's styles are embedded in their `raw` rows
    // rather than set on the row, so only the header appears here.
    var styles: std.ArrayList(theme.Style) = .empty;
    for (rows) |row| if (row.style) |style| try styles.append(a, style);
    try testing.expect(std.mem.indexOfScalar(theme.Style, styles.items, .diff_header) != null);
    try testing.expect(std.mem.indexOfScalar(theme.Style, styles.items, .error_text) != null);
}

test "a running call's dot pulses in the region and settles to the state's colour" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tr = transcript();
    defer tr.deinit();
    const th: theme.Theme = .{ .kind = .truecolor };
    const options: Render = .{ .width = 80, .th = th };
    try tr.apply(.{ .tool_call = .{ .id = 1, .name = "bash", .summary = "Bash(make test)", .detail = "$ make test" } });
    // Running: the dot is lit in one phase and dim in the other, the text the same.
    const lit = try tr.liveRows(a, .{ .width = 80, .th = th, .budget = 10, .pulse = true });
    const dimmed = try tr.liveRows(a, .{ .width = 80, .th = th, .budget = 10, .pulse = false });
    try testing.expectEqualStrings("● Bash(make test)", try stripped(a, lit[0].text));
    try testing.expectEqualStrings("● Bash(make test)", try stripped(a, dimmed[0].text));
    try testing.expect(std.mem.startsWith(u8, lit[0].text, th.paint(.op_running)));
    try testing.expect(std.mem.startsWith(u8, dimmed[0].text, th.paint(.dim)));
    try testing.expect(!std.mem.eql(u8, lit[0].text, dimmed[0].text));
    try testing.expectEqualStrings("└ $ make test", lit[1].text);
    // Nothing has closed: the line is still in flight.
    try testing.expectEqual(@as(usize, 0), (try tr.takeClosed(a, options)).len);
    // The result settles it: a clean run's dot is green and it is written once.
    try tr.apply(.{ .tool_result = .{ .id = 1, .text = "ok\n", .truncated = false, .is_error = false } });
    const closed = try tr.takeClosed(a, options);
    try testing.expectEqual(@as(usize, 2), closed.len);
    try testing.expect(std.mem.startsWith(u8, closed[0].text, th.paint(.op_ok)));
    try testing.expectEqualStrings("● Bash(make test)", try stripped(a, closed[0].text));
    try testing.expectEqualStrings("└ $ make test", closed[1].text);

    // A failed call is red, a write is blue; the ascii set uses a star.
    try tr.apply(.{ .tool_call = .{ .id = 2, .name = "bash", .summary = "Bash(false)" } });
    try tr.apply(.{ .tool_result = .{ .id = 2, .text = "", .truncated = false, .is_error = true, .summary = "exit 1 · 0 lines" } });
    try tr.apply(.{ .tool_call = .{ .id = 3, .name = "write_file", .summary = "Write(a.txt)" } });
    try tr.apply(.{ .tool_result = .{ .id = 3, .text = "created a.txt (6 bytes)", .truncated = false, .is_error = false, .summary = "Wrote 1 line to a.txt" } });
    const more = try tr.takeClosed(a, options);
    try testing.expect(std.mem.startsWith(u8, more[0].text, th.paint(.op_error)));
    try testing.expect(std.mem.startsWith(u8, more[2].text, th.paint(.op_write)));
    try testing.expectEqualStrings("└ Wrote 1 line to a.txt", more[3].text);
    var ascii = transcript();
    defer ascii.deinit();
    try ascii.apply(.{ .tool_call = .{ .id = 1, .name = "read_file", .summary = "Read(x)" } });
    try ascii.apply(.{ .tool_result = .{ .id = 1, .text = "", .truncated = false, .is_error = false } });
    const star = try ascii.takeClosed(a, .{ .width = 80, .th = .{ .kind = .plain, .glyph_set = .ascii } });
    try testing.expectEqualStrings("* Read(x)", try stripped(a, star[0].text));
    // The summary is the model's argument: control bytes never reach the row.
    try ascii.apply(.{ .tool_call = .{ .id = 2, .name = "bash", .summary = "Bash(\x1b[2Jls)" } });
    try ascii.apply(.{ .tool_result = .{ .id = 2, .text = "", .truncated = false, .is_error = false } });
    const clean = try ascii.takeClosed(a, .{ .width = 80, .th = .{ .kind = .plain, .glyph_set = .ascii } });
    try testing.expect(std.mem.indexOf(u8, clean[0].text, "\x1b[2J") == null);
    try testing.expectEqualStrings("* Bash([2Jls)", try stripped(a, clean[0].text));
}

test "a run of tool calls is summed up where the text resumes, and at the end of the turn" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tr = transcript();
    defer tr.deinit();
    const options: Render = .{ .width = 80, .th = .{ .kind = .plain } };
    try tr.apply(.{ .tool_call = .{ .id = 1, .name = "read_file", .summary = "Read(a)" } });
    try tr.apply(.{ .tool_result = .{ .id = 1, .text = "", .truncated = false, .is_error = false } });
    try tr.apply(.{ .tool_call = .{ .id = 2, .name = "read_file", .summary = "Read(b)" } });
    try tr.apply(.{ .tool_result = .{ .id = 2, .text = "", .truncated = false, .is_error = false } });
    try tr.apply(.{ .tool_call = .{ .id = 3, .name = "grep", .summary = "Grep(x)" } });
    try tr.apply(.{ .tool_result = .{ .id = 3, .text = "", .truncated = false, .is_error = false } });
    try tr.apply(.{ .tool_call = .{ .id = 4, .name = "bash", .summary = "Bash(ls)" } });
    try tr.apply(.{ .tool_result = .{ .id = 4, .text = "", .truncated = false, .is_error = false } });
    try tr.apply(.{ .tool_call = .{ .id = 5, .name = "edit_file", .summary = "Edit(a)" } });
    try tr.apply(.{ .tool_result = .{ .id = 5, .text = "", .truncated = false, .is_error = false } });
    try tr.apply(.{ .answer_delta = "Done." });
    try tr.apply(.{ .turn_end = .{ .stop = .eos } });
    const rows = try tr.takeClosed(a, options);
    const s = try texts(a, rows);
    const summary = std.mem.indexOf(u8, s, "Read 2 files, searched 1 time, ran 1 shell command, wrote 1 file").?;
    try testing.expect(summary > std.mem.indexOf(u8, s, "Edit(a)").?);
    try testing.expect(summary < std.mem.indexOf(u8, s, "Done.").?);
    for (rows) |row| if (std.mem.indexOf(u8, row.text, "Read 2 files") != null) try testing.expectEqual(theme.Style.dim, row.style.?);
    // One summary per run: the next turn's counters start empty, and a turn
    // that ends on a call still gets its row.
    tr.reset();
    try tr.apply(.{ .tool_call = .{ .id = 6, .name = "bash", .summary = "Bash(make)" } });
    try tr.apply(.{ .tool_result = .{ .id = 6, .text = "", .truncated = false, .is_error = false } });
    try tr.apply(.{ .turn_end = .{ .stop = .cancelled } });
    const cancelled = try texts(a, try tr.takeClosed(a, options));
    try testing.expect(std.mem.indexOf(u8, cancelled, "Ran 1 shell command") != null);
    try testing.expect(std.mem.indexOf(u8, cancelled, "Read 2") == null);
    // A turn without tools writes nothing of the kind.
    tr.reset();
    try tr.apply(.{ .answer_delta = "Just text." });
    try tr.apply(.{ .turn_end = .{ .stop = .eos } });
    const plain = try texts(a, try tr.takeClosed(a, options));
    try testing.expect(std.mem.indexOf(u8, plain, "shell command") == null);
}

test "a failed tool shows its message under the call, bounded to three rows" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tr = transcript();
    defer tr.deinit();
    const options: Render = .{ .width = 80, .th = .{ .kind = .plain } };
    try tr.apply(.{ .tool_call = .{ .id = 2, .name = "bash", .summary = "Running command", .detail = "$ zig build" } });
    try tr.apply(.{ .tool_result = .{ .id = 2, .text = "error: one\nerror: two\nerror: three\nerror: four\n[bash: exit code 1]", .truncated = false, .is_error = true, .summary = "exit 1 · 4 lines" } });
    const closed = try tr.takeClosed(a, options);
    try testing.expectEqual(@as(usize, 6), closed.len);
    try testing.expectEqualStrings("└ error: one", closed[2].text);
    try testing.expectEqual(theme.Style.error_text, closed[2].style.?);
    try testing.expectEqualStrings("  error: two", closed[3].text);
    try testing.expectEqualStrings("  error: three", closed[4].text);
    try testing.expectEqualStrings("  …", closed[5].text);
}

test "an empty result row is skipped unless the result was truncated" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tr = transcript();
    defer tr.deinit();
    const options: Render = .{ .width = 80, .th = .{ .kind = .plain } };
    try tr.apply(.{ .tool_call = .{ .id = 3, .name = "glob", .summary = "Listing **/*.zig" } });
    try tr.apply(.{ .tool_result = .{ .id = 3, .text = "a.zig\nb.zig", .truncated = true, .is_error = false } });
    const closed = try tr.takeClosed(a, options);
    try testing.expectEqual(@as(usize, 2), closed.len);
    try testing.expectEqualStrings("└ truncated", closed[1].text);
}

test "a diff renders side by side when wide and unified when narrow" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tr = transcript();
    defer tr.deinit();
    const rows = [_]diff.Row{
        .{ .old_line = 1, .new_line = 1, .kind = .context, .text = "keep this" },
        .{ .old_line = 2, .kind = .remove, .text = "old value", .change = .{ .start = 0, .len = 3 } },
        .{ .new_line = 2, .kind = .add, .text = "new value", .change = .{ .start = 0, .len = 3 } },
        .{ .old_line = 3, .new_line = 3, .kind = .context, .text = "keep this too" },
    };
    const wide = blk: {
        try tr.apply(.{ .diff = .{ .path = "a.zig", .rows = &rows } });
        break :blk try texts(a, try tr.takeClosed(a, .{ .width = 120, .th = .{ .kind = .plain } }));
    };
    // Both panes carry the change, the removal left and the addition right.
    const old_at = std.mem.indexOf(u8, wide, "old value") orelse return error.TestUnexpectedResult;
    const new_at = std.mem.indexOf(u8, wide, "new value") orelse return error.TestUnexpectedResult;
    try testing.expect(old_at < new_at);
    // The two-pane separator is present.
    try testing.expect(std.mem.indexOf(u8, wide, "│") != null);

    tr.reset();
    try tr.apply(.{ .diff = .{ .path = "a.zig", .rows = &rows } });
    const narrow = try texts(a, try tr.takeClosed(a, .{ .width = 60, .th = .{ .kind = .plain } }));
    // The unified form shows the change too, but as `-`/`+` rows with no
    // second pane.
    try testing.expect(std.mem.indexOf(u8, narrow, "old value") != null);
    try testing.expect(std.mem.indexOf(u8, narrow, "new value") != null);
    try testing.expect(std.mem.indexOf(u8, narrow, "│") == null);
    try testing.expect(std.mem.indexOf(u8, narrow, "+") != null);
    try testing.expect(std.mem.indexOf(u8, narrow, "-") != null);
}

test "a diff taller than the budget folds under a marker" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tr = transcript();
    defer tr.deinit();
    var rows: std.ArrayList(diff.Row) = .empty;
    for (0..100) |i| {
        try rows.append(a, .{ .old_line = i + 1, .new_line = i + 1, .kind = .add, .text = try std.fmt.allocPrint(a, "added line {d}", .{i}) });
    }
    try tr.apply(.{ .diff = .{ .path = "big.zig", .rows = rows.items } });
    const s = try texts(a, try tr.takeClosed(a, .{ .width = 60, .th = .{ .kind = .plain } }));
    try testing.expect(std.mem.indexOf(u8, s, "… ") != null);
    try testing.expect(std.mem.indexOf(u8, s, "lines") != null);
    try testing.expect(std.mem.indexOf(u8, s, "added line 0") != null);
    try testing.expect(std.mem.indexOf(u8, s, "added line 99") == null);
}

test "the live region shows the tail of what is open, and says how much is above" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tr = transcript();
    defer tr.deinit();
    const th: theme.Theme = .{ .kind = .plain };
    try tr.apply(.{ .thinking_delta = "one\ntwo\nthree\nfour\nfive" });
    const rows = try tr.liveRows(a, .{ .width = 20, .th = th, .budget = 4, .thinking_label = "⠋ thinking… 2s" });
    // Folded: the label alone, because the text is not shown.
    try testing.expectEqual(@as(usize, 1), rows.len);
    try testing.expectEqualStrings("⠋ thinking… 2s", rows[0].text);
    tr.expanded = true;
    const open = try tr.liveRows(a, .{ .width = 20, .th = th, .budget = 4, .thinking_label = "⠋ thinking… 2s" });
    const s = try texts(a, open);
    try testing.expect(std.mem.indexOf(u8, s, "… 4 lines above") != null);
    try testing.expect(std.mem.indexOf(u8, s, "five") != null);
    try testing.expect(std.mem.indexOf(u8, s, "one") == null);
}

test "a step that answers and then calls a tool closes its text; a closed thought never wears the busy label" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tr = transcript();
    defer tr.deinit();
    const options: Render = .{ .width = 40, .th = .{ .kind = .plain } };
    const live: Live = .{ .width = 40, .th = options.th, .budget = 10, .thinking_label = "⠋ thinking… 9s" };

    try tr.apply(.{ .user = "largest radius?" });
    try tr.apply(.{ .thinking_delta = "read it" });
    try tr.apply(.{ .thinking_end = 2.0 });
    try tr.apply(.{ .answer_delta = "The file is long.\n\n" });
    try tr.apply(.{ .tool_call = .{ .id = 1, .name = "bash", .summary = "Running command", .detail = "$ wc -l x" } });
    // The call closed the answer, so the whole step is written; only the running call stays live.
    const written = try texts(a, try tr.takeClosed(a, options));
    try testing.expect(std.mem.indexOf(u8, written, "Thought for 2.0s") != null);
    try testing.expect(std.mem.indexOf(u8, written, "The file is long.") != null);
    try testing.expect(std.mem.indexOf(u8, try texts(a, try tr.liveRows(a, live)), "thinking…") == null);

    try tr.apply(.{ .tool_result = .{ .id = 1, .text = "400 x", .truncated = false, .is_error = false, .summary = "" } });
    try tr.apply(.{ .thinking_delta = "so 400 lines" });
    var rows = try texts(a, try tr.liveRows(a, live));
    try testing.expect(std.mem.indexOf(u8, rows, "⠋ thinking… 9s") != null);
    try tr.apply(.{ .thinking_end = 3.0 });
    // Closed but not yet written: the fold label, not the animated one.
    rows = try texts(a, try tr.liveRows(a, live));
    try testing.expect(std.mem.indexOf(u8, rows, "thinking…") == null);
    try testing.expect(std.mem.indexOf(u8, rows, "Thought for 3.0s") != null);
}

test "a turn that is cancelled mid-thought keeps its text and its bare label" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tr = transcript();
    defer tr.deinit();
    const options: Render = .{ .width = 40, .th = .{ .kind = .plain } };
    tr.expanded = true;
    try tr.apply(.{ .thinking_delta = "half a thought" });
    try tr.apply(.{ .turn_end = .{ .stop = .cancelled } });
    const s = try texts(a, try tr.takeClosed(a, options));
    try testing.expect(std.mem.indexOf(u8, s, "▾ thinking (Tab to fold)") != null);
    try testing.expect(std.mem.indexOf(u8, s, "half a thought") != null);
    try testing.expect(std.mem.indexOf(u8, s, "— cancelled") != null);
}

test "reset drops the blocks and the counters, not the scrollback" {
    var tr = transcript();
    defer tr.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const options: Render = .{ .width = 40, .th = .{ .kind = .plain } };
    try tr.apply(.{ .user = "one" });
    _ = try tr.takeClosed(arena.allocator(), options);
    try testing.expect(tr.rows > 0);
    tr.reset();
    try testing.expectEqual(@as(usize, 0), tr.rows);
    try testing.expectEqual(@as(usize, 0), tr.written);
    try testing.expect(!tr.pending());
}
