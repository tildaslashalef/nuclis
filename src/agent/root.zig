//! `nuclis agent`: the interactive surface of the inference engine. One
//! conversation streamed into the terminal's scrollback, folding thinking,
//! and a statistics bar with the same timing definitions as `bench`.
//!
//! `src/agent/` is the composition layer: it owns the engine, the
//! conversation, the session log, and the tool loop, and drives the
//! engine-free terminal surface in `src/tui/`. The boundary is one-way —
//! `src/tui/` never imports `inference` — which lets the surface be tested
//! without a model and the engine without a terminal. Phase 2's design is in
//! docs/agent-spec.md.
//!
//! Rendering model: the agent does not own a full-screen cell grid. Startup
//! clears the visible screen (scrollback preserved), prints a header, and
//! anchors a small live region at the bottom; completed turns are written
//! once above it and left to scroll away. This file decides *what* each row
//! says; `tui.screen` owns how rows reach the terminal.
//!
//! Multi-turn without replay: `loop.Completer` owns the consumed text; the
//! next turn prefills only the remainder when the render extends it, and
//! resets and replays otherwise. Recurrent state is never rewound by hand.
const std = @import("std");
const inference = @import("inference");
const engine = @import("../engine.zig");
const generate = @import("../generate.zig");
const config = @import("../config.zig");
const catalog = @import("../catalog.zig");
const version = @import("build_options").version;
const paths = @import("../paths.zig");
const model = @import("../model.zig");
const interrupt = @import("../interrupt.zig");
const tui = @import("../tui/root.zig");
const prompt_history = @import("history.zig");
const session_log = @import("session.zig");
const commands = @import("commands.zig");
pub const resume_mod = @import("resume.zig");
pub const print_mode = @import("print.zig");
pub const tools = @import("tools/root.zig");
pub const loop = @import("loop.zig");
const stream = @import("stream.zig");
const terminal = tui.terminal;
const screen = tui.screen;
const editor = tui.editor;
const transcript = tui.transcript;
const status_bar = tui.status;
const view = tui.view;
const keys = tui.keys;
const theme = tui.theme;
const markdown = tui.markdown;
const choice = tui.choice;
const Profile = inference.profiles;

/// Idle height of the highlighted input box, in rows.
const min_editor_rows = 3;

/// One paintable row. The type is the screen module's: the agent builds
/// rows, `tui.screen` decides how they reach the terminal.
const Row = screen.Row;

/// Per-turn statistics with `bench`'s definitions: prefill covers this turn's
/// new prompt tokens; decode covers `generated - 1` steps after the first token.
const Stats = struct {
    prompt_tokens: usize = 0,
    generated: usize = 0,
    prefill_seconds: f64 = 0,
    decode_seconds: f64 = 0,
    replayed: bool = false,

    fn prefillRate(self: Stats) ?f64 {
        return if (self.prompt_tokens > 0 and self.prefill_seconds > 0) @as(f64, @floatFromInt(self.prompt_tokens)) / self.prefill_seconds else null;
    }
    fn decodeRate(self: Stats) ?f64 {
        return if (self.generated > 1 and self.decode_seconds > 0) @as(f64, @floatFromInt(self.generated - 1)) / self.decode_seconds else null;
    }
};

const Ui = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    eng: *engine.Engine,
    term: *terminal.Terminal,
    /// The live region: every row the agent paints goes through it.
    scr: *screen.Screen,
    th: theme.Theme,
    /// The agent loop: owns the conversation history, the step budget, and the
    /// tool registry. The driver presents its events and feeds it input.
    agent: *loop.Agent,
    /// The loop's completion side, held so the main loop can point it at a
    /// re-opened engine and refresh the effort between turns.
    completer: *loop.Completer,
    /// The prompt editor: buffer, cursor, paste chips, prompt history.
    ed: *editor.Editor,
    /// The turn as blocks: what the events build and the screen shows.
    tr: *transcript.Transcript,
    /// The session file: append-only, one entry per turn boundary.
    log: *session_log.Session,
    /// The completion list, empty unless the word under the cursor asks for
    /// one (`/command`, `@path`).
    comp: *choice.Choice,
    comp_kind: enum { none, command, path } = .none,
    effort: Profile.Effort,
    /// Configured and flagged sampling overrides, applied over the effort's
    /// profile per turn.
    overrides: Profile.SamplingOverrides,
    /// The artifact's profile (the engine's, selected by template digest):
    /// the sampling defaults per effort.
    profile: Profile.Profile,
    /// Tokens the session has consumed, for the sampler's penalties.
    tokens_seen: *inference.sampling.History,
    /// The instrumented bar, fed by the turn's status events.
    bar: status_bar.Status = .{},
    /// The model's short name for the bar: the name it was reached through,
    /// else the artifact's own.
    model_label: []const u8 = "",
    busy: bool = false,
    quit: bool = false,
    /// Whether the terminal has focus; a turn that ends unfocused notifies.
    focused: bool = true,
    /// Whether desktop notifications are enabled (`NUCLIS_NO_NOTIFY`).
    notify: bool = true,
    status: []const u8 = "ready",
    stats: Stats = .{},
    turn_started: std.Io.Timestamp,
    first_token: ?std.Io.Timestamp = null,
    /// The step's prefill as the beats report it: when its first beat arrived
    /// and at what position, and the rate measured from there. The rate
    /// outlives the prefill so the bar shows it beside the decode rate.
    prefill_started: ?std.Io.Timestamp = null,
    prefill_base: usize = 0,
    prefill_rate: ?f64 = null,
    submit: bool = false,
    /// Spinner frame, advanced on every paint while a turn runs.
    frame: usize = 0,
    /// Incomplete key sequence carried between reads.
    pending: [16]u8 = undefined,
    pending_len: usize = 0,
    /// Set after a Ctrl-C cancel so a second press during the same turn quits.
    cancel_pending: bool = false,
    /// Context size requested with Ctrl-W; the main loop re-opens the engine.
    ctx_request: ?usize = null,
    /// Set the first time a session write failed, so the notice is printed
    /// once rather than after every turn.
    log_failed: bool = false,
    /// Ctrl-N asks the main loop for a fresh session file.
    new_session: bool = false,
    /// A message typed while a turn was running, to be sent as the next one
    /// (steering). Owned.
    queued: std.ArrayList(u8) = .empty,
    /// An open picker owns the keyboard while it is up; `none` otherwise.
    picker: enum { none, resume_session } = .none,
    /// The paths the open picker offers, parallel to `comp`'s items. Owned.
    resume_paths: std.ArrayList([]u8) = .empty,
    /// A session path to replay on the next main-loop pass, set by the picker
    /// and consumed there because the loop owns the session file. Owned.
    resume_request: ?[]u8 = null,

    fn deinit(self: *Ui) void {
        self.queued.deinit(self.alloc);
        for (self.resume_paths.items) |path| self.alloc.free(path);
        self.resume_paths.deinit(self.alloc);
        if (self.resume_request) |path| self.alloc.free(path);
        if (self.model_label.len != 0) self.alloc.free(self.model_label);
    }
    fn newSession(self: *Ui) void {
        // The log is replaced by the main loop, which owns its storage.
        self.new_session = true;
        self.tr.reset();
        self.bar = .{};
        self.agent.resetConversation();
        self.completer.reset();
        self.eng.model.reset();
        self.tokens_seen.reset();
        self.stats = .{};
        self.status = "new session";
    }

    // ----- drawing -----

    /// Drops rows from the top so the live region never exceeds the screen.
    fn clampTop(rows: *std.ArrayList(Row), max: usize) void {
        if (rows.items.len <= max) return;
        const drop = rows.items.len - max;
        std.mem.copyForwards(Row, rows.items[0..max], rows.items[drop..]);
        rows.shrinkRetainingCapacity(max);
    }

    /// A full-width separator rule, from the theme's glyph set (one cell per
    /// repetition in both tables).
    fn rule(a: std.mem.Allocator, cells_w: usize, th: theme.Theme) ![]const u8 {
        const separator = th.glyphs().rule;
        const bytes = try a.alloc(u8, separator.len * cells_w);
        var i: usize = 0;
        while (i < cells_w) : (i += 1) @memcpy(bytes[separator.len * i ..][0..separator.len], separator);
        return bytes;
    }

    /// The bar's value for this frame: the measured state the events left in
    /// `bar`, plus the settings only the agent knows (the context window, the
    /// effort, the word on the left). `tui.status` owns the layout.
    fn statusLine(self: *Ui, a: std.mem.Allocator, columns: usize) ![]const u8 {
        const session = self.eng.model.session();
        var bar = self.bar;
        bar.busy = self.busy;
        bar.message = self.status;
        bar.context_used = session.position;
        bar.context_capacity = session.capacity;
        bar.effort = @tagName(self.effort);
        bar.model = self.model_label;
        if (self.busy) {
            bar.prompt_tokens = self.stats.prompt_tokens;
            bar.generated = self.stats.generated;
            bar.replayed = self.stats.replayed;
            bar.step = @min(self.agent.steps_done + 1, self.agent.budget);
            bar.budget = self.agent.budget;
        }
        return bar.paint(a, .{
            .width = columns,
            .th = self.th,
            .frame = self.frame,
            .elapsed_seconds = seconds(self.turn_started, std.Io.Clock.awake.now(self.io)),
        });
    }

    fn draw(self: *Ui) !void {
        const size = self.term.size();
        // A resize invalidates the width every printed row was wrapped at.
        // Completed turns are immutable (agent-spec § Rendering model): only
        // the last one is replayed at the new width, older ones are left as
        // the terminal reflowed them, and the region is rebuilt below.
        const resized = self.scr.resized(size);
        var arena = std.heap.ArenaAllocator.init(self.alloc);
        defer arena.deinit();
        const a = arena.allocator();
        if (resized) try self.replayTurn(a);
        // Whatever closed since the last frame belongs above the region, not
        // in it: a long answer moves into the scrollback block by block while
        // it is written.
        try self.commit(a);
        var rows: std.ArrayList(Row) = .empty;
        const columns = self.columnsFor();
        const editor_cap = @min(@as(usize, 6), size.rows -| 5);
        // The completion list sits between the turn and the input box, inside
        // the region like every other overlay: nuclis has no alternate screen.
        const list = try self.comp.layout(a, .{
            .width = columns -| 3,
            .max_rows = @min(@as(usize, 6), size.rows / 4),
            .th = self.th,
        });
        // The streaming block gets the screen minus editor rows, the list,
        // the spacer, the hint, rule, and bar, and one spare line. Saturated
        // twice rather than `rows -| (cap + 5)`: the combined form hit a
        // spurious integer-overflow panic in Zig 0.16.0 (aarch64) on plain
        // values; this form is semantically identical.
        const max_live = ((size.rows -| editor_cap) -| list.len) -| 5;
        if (self.busy) {
            const live = try self.tr.liveRows(a, .{
                .width = columns,
                .th = self.th,
                .budget = @max(max_live, 1),
                .thinking_label = try self.busyThinkingLabel(a),
                .spinner = self.spinnerFrame(),
            });
            try rows.appendSlice(a, live);
            clampTop(&rows, max_live);
        }
        try rows.appendSlice(a, list);
        // One blank row always separates the streaming answer from the input
        // box, so output is never crammed against what is being typed.
        try rows.append(a, .{ .text = "" });
        const editor_base = rows.items.len;
        const layout = try self.ed.layout(a, .{
            .width = columns -| 3,
            .min_rows = min_editor_rows,
            .max_rows = editor_cap,
            .theme = self.th,
        });
        try rows.appendSlice(a, layout.rows);
        // Only the few shortcuts worth remembering sit under the input box;
        // the full key list is one `/help` away, so the line never overruns.
        const hint = if (self.th.glyph_set == .ascii)
            "Enter send | Shift-Enter newline | up/down history | Tab complete | /help"
        else
            "Enter send · Shift-Enter newline · ↑↓ history · Tab complete · /help";
        const hint_rows = try view.lines(a, hint, columns - 1, .character);
        try rows.append(a, .{ .text = hint_rows[0], .style = .dim });
        try rows.append(a, .{ .text = try rule(a, columns, self.th), .style = .dim });
        try rows.append(a, .{ .text = try self.statusLine(a, columns), .bar = true, .raw = true });
        if (self.busy) self.frame += 1;
        // The cursor is parked after the editor's insertion point (two cells
        // of prompt plus one for the 1-based column) whether or not a turn is
        // running: since step 9 the editor stays live while the model works,
        // so hiding the cursor would hide where the next message is being
        // typed.
        try self.scr.paint(rows.items, .{
            .row = editor_base + layout.cursor_row,
            .column = 3 + layout.cursor_col,
        });
    }

    /// Writes the blocks that closed since the last frame above the live
    /// region. The transcript decides what is closed and hands over each row
    /// exactly once; this function only carries them to the screen.
    fn commit(self: *Ui, a: std.mem.Allocator) !void {
        if (!self.tr.pending()) return;
        const rows = try self.tr.takeClosed(a, .{ .width = self.columnsFor(), .th = self.th });
        if (rows.len > 0) try self.scr.insertAbove(rows);
    }

    /// Rewrites the current turn's rows in place: the only transcript
    /// rewriting the agent does (a resize, a fold toggle). Older turns keep
    /// the width and the fold state they were printed with.
    ///
    /// A rewrite can only reach rows that are still on the screen, so a turn
    /// taller than the space above the region is left alone rather than
    /// half-rewritten — the fold state still applies to whatever is printed
    /// next; before this rule the walk was silently clamped at the top
    /// row and the rewrite landed on the wrong lines.
    fn replayTurn(self: *Ui, a: std.mem.Allocator) !void {
        const replacing = self.tr.rows;
        if (replacing == 0) return;
        if (replacing + self.scr.region_rows >= self.scr.size.rows) return;
        const rows = try self.tr.replayRows(a, .{ .width = self.columnsFor(), .th = self.th });
        try self.scr.replaceAbove(rows, replacing);
    }

    /// Build width for every row, live or transcript: the screen's width
    /// with one cell of margin kept clear, so a full row never wraps and
    /// breaks the cursor arithmetic. The screen owns the size, refreshed by
    /// `draw`, so a row built here and a row painted there agree.
    fn columnsFor(self: *Ui) usize {
        return @max(self.scr.size.columns -| 1, 10);
    }

    /// Applies an event and writes out whatever it closed, at once. Used
    /// where a row must be on the screen before the next thing can fail —
    /// the committed prompt, a notice under it — rather than at the next
    /// frame.
    fn emit(self: *Ui, e: tui.event.Event) !void {
        try self.tr.apply(e);
        var arena = std.heap.ArenaAllocator.init(self.alloc);
        defer arena.deinit();
        try self.commit(arena.allocator());
    }

    /// Appends one entry to the session file. A log write never costs a turn:
    /// the conversation continues and the reason is said once, dimly, the
    /// first time it happens (retention and storage are the user's, so a full
    /// disk is not a reason to lose what the model just wrote).
    fn record(self: *Ui, entry: session_log.Entry) void {
        var stamp: [20]u8 = undefined;
        const now = model.rfc3339(&stamp, std.Io.Timestamp.now(self.io, .real).toSeconds());
        self.log.append(entry, now) catch |err| {
            if (self.log_failed) return;
            self.log_failed = true;
            var note: [128]u8 = undefined;
            self.emit(.{ .notice = std.fmt.bufPrint(&note, "  — session not recorded: {s}", .{@errorName(err)}) catch "  — session not recorded" }) catch {};
        };
    }

    /// The sink the loop drives.
    fn events(self: *Ui) loop.Events {
        return .{ .context = self, .send = eventSend, .record = eventRecord };
    }

    /// The loop's display sink: every event lands in the transcript, and the
    /// turn boundary also settles the bar. The high-frequency progress beats
    /// arrive through `onProgress` instead, because they repaint.
    pub fn send(self: *Ui, e: tui.event.Event) !void {
        try self.tr.apply(e);
        switch (e) {
            .status, .turn_end => self.bar.apply(e),
            else => {},
        }
    }

    fn eventSend(context: *anyopaque, e: tui.event.Event) anyerror!void {
        const self: *Ui = @ptrCast(@alignCast(context));
        try self.send(e);
    }

    /// The loop's session sink: one `Record` becomes one append-only entry.
    fn eventRecord(context: *anyopaque, entry: loop.Record) anyerror!void {
        const self: *Ui = @ptrCast(@alignCast(context));
        var call: [16]u8 = undefined;
        const mapped: session_log.Entry = switch (entry) {
            .user => |text| .{ .user = .{ .text = text } },
            .assistant => |step| .{ .assistant = .{
                .thinking = step.thinking,
                .answer = step.answer,
                .tool_calls = step.calls,
                .stop = @tagName(step.stop),
                .stats = .{
                    .prompt_tokens = step.prompt_tokens,
                    .generated = step.generated,
                    .prefill_seconds = step.prefill_seconds,
                    .decode_seconds = step.decode_seconds,
                    .thinking_seconds = step.thinking_seconds,
                    .replayed = step.replayed,
                },
            } },
            .compaction => |c| .{ .compaction = .{ .first_kept = c.first_kept, .reason = c.reason } },
            .tool_result => |result| .{ .tool_result = .{
                .call = std.fmt.bufPrint(&call, "{d}", .{result.call}) catch "?",
                .text = result.text,
                .truncated = result.truncated,
                .is_error = result.is_error,
                .summary = result.summary,
            } },
        };
        self.record(mapped);
    }

    // ----- input -----

    /// Reads pending keys. While a turn runs only cancellation and folding
    /// are honoured; editing keys are ignored rather than queued. Bytes that
    /// end mid-sequence (split escape or UTF-8) wait in `pending` for the
    /// next read.
    fn poll(self: *Ui, timeout: i32) !void {
        // Large enough that a paste of a few KB takes a handful of reads,
        // each followed by one repaint, rather than dozens.
        var bytes: [4096]u8 = undefined;
        const held = self.pending_len;
        @memcpy(bytes[0..held], self.pending[0..held]);
        const n = held + (self.term.read(bytes[held..], timeout) catch |err| switch (err) {
            error.EndOfStream => {
                self.quit = true;
                return;
            },
            else => return err,
        });
        var i: usize = 0;
        while (i < n) {
            const decoded = keys.next(bytes[i..n]) orelse break;
            i += decoded.consumed;
            try self.handle(decoded.key);
        }
        // Keep one incomplete sequence for the next read; anything longer
        // than a sequence can be is garbage.
        const rest = bytes[i..n];
        self.pending_len = if (rest.len <= self.pending.len) rest.len else 0;
        if (self.pending_len > 0) @memcpy(self.pending[0..rest.len], rest);
    }
    /// Routes one key. The editor owns editing, history recall, and the
    /// paste protocol (`tui.editor`); this function owns only the keys that
    /// mean something to a *session* — cancel, quit, effort, context, fold —
    /// which is the same split as the module boundary.
    fn handle(self: *Ui, key: keys.Key) !void {
        // Focus reports arrive whatever the state, and go to no one else.
        switch (key) {
            .focus_in => {
                self.focused = true;
                return;
            },
            .focus_out => {
                self.focused = false;
                return;
            },
            else => {},
        }
        if (self.picker != .none) {
            // An open picker owns the keyboard: it moves, accepts, or is
            // dismissed; nothing is typed into the editor beneath it.
            switch (key) {
                .up => self.comp.move(.previous),
                .down => self.comp.move(.next),
                .enter, .tab => try self.acceptResume(),
                .ctrl => |c| {
                    if (c == 'c') self.closePicker();
                },
                else => {},
            }
            return;
        }
        if (self.busy) {
            // Steering: the editor stays live while the model
            // works, so the next message can be composed — and sent — without
            // waiting. Only cancellation is the agent's while a turn runs;
            // everything else is editing, including a paste.
            if (!self.ed.pasting) {
                switch (key) {
                    .ctrl => |c| switch (c) {
                        'c' => {
                            // First press cancels the turn; a second quits.
                            interrupt.request();
                            if (self.cancel_pending) self.quit = true else self.cancel_pending = true;
                            return;
                        },
                        'd' => {
                            interrupt.request();
                            self.quit = true;
                            return;
                        },
                        else => {},
                    },
                    else => {},
                }
            }
            switch (try self.ed.handleKey(key)) {
                // Enter queues rather than sends: the turn in flight finishes
                // first, and the message goes out as the next one.
                .submit => try self.queue(),
                .none => {},
                .ignored => if (key == .tab) {
                    self.tr.expanded = !self.tr.expanded;
                },
            }
            try self.refreshCompletion();
            return;
        }
        // A completion list takes the keys that drive it, and only those.
        if (!self.comp.isEmpty()) {
            switch (key) {
                .up => return self.comp.move(.previous),
                .down => return self.comp.move(.next),
                .tab => return self.accept(),
                else => {},
            }
        }
        switch (try self.ed.handleKey(key)) {
            .none => return self.refreshCompletion(),
            .submit => {
                self.submit = true;
                try self.comp.set(&.{});
                return;
            },
            .ignored => {},
        }
        switch (key) {
            .ctrl => |c| switch (c) {
                'd' => self.quit = true,
                'c' => {
                    self.quit = true;
                    var farewell: [16]u8 = undefined;
                    try self.scr.insertAbove(&.{.{ .text = std.fmt.bufPrint(&farewell, "bye {s}", .{self.th.glyphs().effort}) catch "bye", .style = .dim }});
                },
                't' => {
                    self.effort = switch (self.effort) {
                        .off => .low,
                        .low => .medium,
                        .medium => .high,
                        .high => .xhigh,
                        .xhigh => .off,
                    };
                    self.record(.{ .effort = .{ .effort = @tagName(self.effort) } });
                },
                'n' => self.newSession(),
                'w' => {
                    // Cycle the context window; the main loop re-opens the
                    // engine because KV capacity is allocated at open time.
                    const sizes = [_]usize{ 2048, 4096, 8192, 16384, 32768 };
                    const current = self.eng.model.session().capacity;
                    self.ctx_request = for (sizes) |s| {
                        if (s > current) break s;
                    } else sizes[0];
                },
                else => {},
            },
            .tab => {
                self.tr.expanded = !self.tr.expanded;
                // The last turn is rewritten with the new state so unfolding
                // works after the answer is already on screen; turns that
                // scrolled away keep what they were printed with.
                var arena = std.heap.ArenaAllocator.init(self.alloc);
                defer arena.deinit();
                try self.replayTurn(arena.allocator());
            },
            else => {},
        }
    }

    /// Keeps a message typed during a turn for the next one. The prompt is
    /// shown in the transcript only when it is actually sent, so a queued
    /// message can still be edited — it goes back into the editor untouched
    /// if the user quits before the turn ends.
    fn queue(self: *Ui) !void {
        if (self.ed.isEmpty()) return;
        self.queued.clearRetainingCapacity();
        try self.queued.appendSlice(self.alloc, self.ed.text());
        self.ed.clear();
        try self.comp.set(&.{});
        self.status = "queued";
    }

    /// Opens the `/resume` picker from the sessions of this workspace. The
    /// list is emptied and an explanatory notice is left when there is nothing
    /// to choose, so a dead end always says why.
    fn openResumePicker(self: *Ui, root_dir: ?[]const u8, cwd: []const u8) !void {
        const root = root_dir orelse {
            try self.emit(.{ .notice = "  — no user root: cannot resume" });
            return;
        };
        const summaries = resume_mod.list(self.alloc, self.io, root, cwd) catch |err| {
            var note: [96]u8 = undefined;
            try self.emit(.{ .notice = std.fmt.bufPrint(&note, "  — {s} listing sessions", .{@errorName(err)}) catch "  — cannot list sessions" });
            return;
        };
        defer resume_mod.freeList(self.alloc, summaries);
        if (summaries.len == 0) {
            try self.emit(.{ .notice = "  — no saved sessions for this workspace" });
            return;
        }
        errdefer self.closePicker();
        var arena = std.heap.ArenaAllocator.init(self.alloc);
        defer arena.deinit();
        const a = arena.allocator();
        var items: std.ArrayList(choice.Item) = .empty;
        for (summaries) |summary| {
            try self.resume_paths.append(self.alloc, try self.alloc.dupe(u8, summary.path));
            const short = if (summary.id.len > 8) summary.id[0..8] else summary.id;
            try items.append(a, .{
                .label = try std.fmt.allocPrint(a, "{s}  {s}", .{ short, summary.time }),
                .detail = if (summary.first_prompt.len > 0)
                    try std.fmt.allocPrint(a, "{s} · ctx {d} · {s}", .{ summary.effort, summary.ctx_size, summary.first_prompt })
                else
                    try std.fmt.allocPrint(a, "{s} · ctx {d} · {s}", .{ summary.effort, summary.ctx_size, std.fs.path.basename(summary.cwd) }),
            });
        }
        try self.comp.set(items.items);
        self.picker = .resume_session;
        self.status = "resume";
    }

    /// Dismisses the picker and drops what it offered. A resumed session is
    /// not applied here: the picker only names a path, and the main loop —
    /// which owns the session file — replays it.
    fn closePicker(self: *Ui) void {
        self.picker = .none;
        for (self.resume_paths.items) |path| self.alloc.free(path);
        self.resume_paths.clearRetainingCapacity();
        self.comp.set(&.{}) catch {};
    }

    fn acceptResume(self: *Ui) !void {
        if (self.comp.selected >= self.resume_paths.items.len) return;
        self.resume_request = try self.alloc.dupe(u8, self.resume_paths.items[self.comp.selected]);
        self.closePicker();
    }

    /// Rebuilds the completion list from the word under the cursor: the
    /// commands when it is a `/word` at the start of the input, the workspace
    /// entries when it is an `@path`, and nothing otherwise. Rebuilt on every
    /// key rather than opened and closed, so the list can never disagree with
    /// what is typed.
    fn refreshCompletion(self: *Ui) !void {
        const word = self.ed.wordBefore();
        var arena = std.heap.ArenaAllocator.init(self.alloc);
        defer arena.deinit();
        const a = arena.allocator();
        var items: std.ArrayList(choice.Item) = .empty;
        if (word.len >= 1 and word[0] == '/' and self.ed.atFirstWord()) {
            self.comp_kind = .command;
            var buffer: [commands.table.len]commands.Spec = undefined;
            for (commands.matching(word[1..], &buffer)) |spec| {
                const label = if (spec.argument.len > 0)
                    try std.fmt.allocPrint(a, "/{s} {s}", .{ spec.name, spec.argument })
                else
                    try std.fmt.allocPrint(a, "/{s}", .{spec.name});
                try items.append(a, .{ .label = label, .detail = spec.summary });
            }
        } else if (word.len >= 1 and word[0] == '@') {
            self.comp_kind = .path;
            // The workspace is the process's working directory, always.
            for (try commands.workspacePaths(a, self.io, .cwd(), word[1..])) |path| {
                try items.append(a, .{ .label = try std.fmt.allocPrint(a, "@{s}", .{path}) });
            }
        } else {
            self.comp_kind = .none;
        }
        try self.comp.set(items.items);
    }

    /// Takes the highlighted completion into the editor. A command keeps only
    /// its name (the argument sketch is a hint, not text), and gains a space
    /// so the argument can be typed straight away; a directory keeps its
    /// separator, so the next Tab walks into it.
    fn accept(self: *Ui) !void {
        const item = self.comp.current() orelse return;
        const label = item.label[0 .. std.mem.indexOfScalar(u8, item.label, ' ') orelse item.label.len];
        try self.ed.replaceWord(label);
        if (self.comp_kind == .command) try self.ed.insert(" ");
        try self.refreshCompletion();
    }

    /// The spinner frame for this repaint, from the theme's glyph set. The
    /// same frame drives the thinking label and a running tool call.
    fn spinnerFrame(self: *Ui) []const u8 {
        const frames = self.th.glyphs().spinner;
        return frames[self.frame % frames.len];
    }

    /// The label of a thinking block that has not closed yet: an animated dot
    /// and counting seconds. It is the agent's because it reads a clock and a
    /// spinner frame, which the transcript has neither of; every *closed*
    /// label is the transcript's and carries the measured time.
    fn busyThinkingLabel(self: *Ui, a: std.mem.Allocator) ![]const u8 {
        var w: std.Io.Writer.Allocating = .init(a);
        const gl = self.th.glyphs();
        const hint = if (self.tr.expanded) "Tab to fold" else "Tab to unfold";
        const spin = gl.spinner[self.frame % gl.spinner.len];
        try w.writer.print("{s} thinking{s} {d:.0}s ({s})", .{ spin, gl.ellipsis, seconds(self.turn_started, std.Io.Clock.awake.now(self.io)), hint });
        return w.written();
    }

    // ----- generation hooks -----

    /// The engine's turn beat: after every prefill chunk and
    /// every generated token. It replaces the loop's per-token `step` hook,
    /// which fired once for a whole chunked prefill and left the spinner
    /// frozen through a 15-second prompt while a counter guessed one token
    /// per call. The position here is measured, not counted.
    fn onProgress(context: *anyopaque, value: inference.observer.Progress) !void {
        const self: *Ui = @ptrCast(@alignCast(context));
        const now = std.Io.Clock.awake.now(self.io);
        if (value.phase == .decode) {
            if (self.first_token == null) self.first_token = now;
            self.prefill_started = null;
            self.status = "generating";
            self.stats.generated = value.position;
            self.bar.generated = value.position;
        } else {
            // A step's prefill target is the prompt it is consuming; the bar
            // counts down from it until the first token arrives. The rate is
            // measured from the step's first beat, not the turn's start, so a
            // later step's prefill is not diluted by the decode before it.
            self.stats.prompt_tokens = value.target;
            if (self.prefill_started) |started| {
                const elapsed = seconds(started, now);
                if (value.position > self.prefill_base and elapsed > 0) self.prefill_rate = @as(f64, @floatFromInt(value.position - self.prefill_base)) / elapsed;
            } else {
                // A new step: its decode rate starts from its own first
                // token, and the word on the left says what is happening.
                self.prefill_started = now;
                self.prefill_base = value.position;
                self.first_token = null;
                self.status = "prefill";
            }
        }
        self.bar.apply(.{ .status = .{
            .phase = switch (value.phase) {
                .prefill => .prefill,
                .decode => .decode,
            },
            .position = value.position,
            .target = value.target,
            .rates = self.liveRates(),
            .elapsed_ns = @intCast(self.turn_started.durationTo(std.Io.Clock.awake.now(self.io)).nanoseconds),
        } });
        try self.poll(0);
        try self.draw();
    }

    /// What the turn has measured so far: an elapsed-time estimate while it
    /// runs, and nothing at all before there is something to divide by. The
    /// measured values replace these when the turn ends (`turn_end`).
    fn liveRates(self: *Ui) tui.event.Rates {
        const now = std.Io.Clock.awake.now(self.io);
        var rates: tui.event.Rates = .{ .prefill = self.prefill_rate };
        if (self.first_token) |first| {
            const elapsed = seconds(first, now);
            if (self.stats.generated > 1 and elapsed > 0) rates.decode = @as(f64, @floatFromInt(self.stats.generated - 1)) / elapsed;
        }
        return rates;
    }
};

/// Prefills the system block and tools so the first turn pays only its own
/// message, shown as a warm-up in the bar (Enter queues meanwhile). A cancel
/// or a window too small leaves the session unprimed with a notice; the
/// turn itself then reports what does not fit.
fn primeSession(ui: *Ui) void {
    ui.completer.effort = ui.effort;
    ui.busy = true;
    ui.status = "warming up";
    ui.turn_started = std.Io.Clock.awake.now(ui.io);
    ui.first_token = null;
    ui.prefill_started = null;
    ui.prefill_rate = null;
    ui.bar = .{ .phase = .prefill };
    ui.draw() catch {};
    defer {
        ui.busy = false;
        ui.status = "ready";
        ui.bar = .{};
        ui.stats = .{};
        interrupt.clear();
    }
    _ = ui.completer.prime(ui.agent.system, ui.agent.tool_defs) catch |err| {
        var note: [192]u8 = undefined;
        const text = if (err == error.ContextFull) blk: {
            const overflow = ui.completer.overflow orelse loop.Overflow{ .needed = 0, .capacity = ui.eng.model.session().capacity };
            break :blk std.fmt.bufPrint(&note, "  — context window too small for the system prompt and tools: {d} tokens (prefix plus output budget) of {d}; raise it with /ctx <n>", .{ overflow.needed, overflow.capacity }) catch "  — context window too small for the system prompt and tools";
        } else std.fmt.bufPrint(&note, "  — warm-up skipped: {s}", .{@errorName(err)}) catch "  — warm-up skipped";
        ui.emit(.{ .notice = text }) catch {};
    };
}

fn seconds(from: std.Io.Timestamp, to: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(from.durationTo(to).nanoseconds)) / std.time.ns_per_s;
}

/// The tool-tick seam: a tool that already polls (`bash`) calls this so
/// the driver reads the keyboard and repaints while the command holds the
/// loop. A Ctrl-C read here sets the interrupt the tool consults, which closes
/// the cancel gap a blocking command otherwise leaves open. Errors are
/// swallowed by design: a frame that cannot be drawn is not a reason to fail
/// a tool.
fn toolTick(context: *anyopaque) void {
    const self: *Ui = @ptrCast(@alignCast(context));
    self.poll(0) catch {};
    self.draw() catch {};
}

/// Runs one user turn through the loop: prepares the display state, hands the
/// prompt to the agent, and notifies once it is done if the window was
/// elsewhere. The loop owns everything the model saw and produced.
fn runTurn(ui: *Ui, sampler: *inference.sampling.Sampler, user: []const u8) !void {
    // The effort may have changed since the last turn (Ctrl-T): rebuild the
    // sampling options from its profile, keeping the RNG's state.
    ui.completer.effort = ui.effort;
    try sampler.setOptions(ui.profile.samplingOptions(ui.effort, ui.overrides));
    // A new turn starts a new set of blocks; the previous turn's rows stay in
    // the terminal's scrollback, where they belong.
    ui.tr.reset();
    ui.busy = true;
    ui.cancel_pending = false;
    ui.first_token = null;
    ui.prefill_started = null;
    ui.prefill_rate = null;
    ui.turn_started = std.Io.Clock.awake.now(ui.io);
    ui.stats = .{};
    ui.bar = .{ .phase = .prefill };
    ui.status = "prefill";
    interrupt.clear();
    defer ui.busy = false;
    const stop = try ui.agent.turn(user);
    ui.status = switch (stop) {
        .done => "ready",
        .budget => "step budget",
        .cancelled => "cancelled",
    };
    if (terminal.shouldNotify(ui.focused, ui.notify, stop != .cancelled)) {
        try ui.term.notify("nuclis: response ready");
    }
}

/// Executes a slash command. Everything it can do, a key can do too (the
/// table in `commands.zig` says which); what it adds is a value — a context
/// size that is not on the cycle, an effort by name — and `/save`, which has
/// no key at all. Every outcome is a dim notice in the transcript, so the
/// conversation records what was asked of the agent as well as of the model.
fn runCommand(ui: *Ui, parsed: commands.Result, root_dir: ?[]const u8, cwd: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(ui.alloc);
    defer arena.deinit();
    const a = arena.allocator();
    switch (parsed) {
        .unknown => |word| {
            var names: std.ArrayList(u8) = .empty;
            for (commands.table, 0..) |spec, i| {
                if (i > 0) try names.appendSlice(a, ", ");
                try names.append(a, '/');
                try names.appendSlice(a, spec.name);
            }
            try ui.emit(.{ .notice = try std.fmt.allocPrint(a, "  — /{s} is not a command; known: {s}", .{ word, names.items }) });
        },
        .usage => |spec| try ui.emit(.{ .notice = try std.fmt.allocPrint(a, "  — usage: /{s} {s} — {s}", .{ spec.name, spec.argument, spec.summary }) }),
        .command => |command| switch (command) {
            .help => {
                const rows = try commands.help(a, ui.th.glyph_set == .ascii);
                try ui.emit(.{ .info = try std.mem.join(a, "\n", rows) });
            },
            .new => {
                ui.newSession();
                try ui.emit(.{ .notice = "  — new session" });
            },
            .resume_session => try ui.openResumePicker(root_dir, cwd),
            .think => |name| {
                if (std.meta.stringToEnum(Profile.Effort, name)) |effort| {
                    ui.effort = effort;
                    ui.record(.{ .effort = .{ .effort = @tagName(effort) } });
                    try ui.emit(.{ .notice = try std.fmt.allocPrint(a, "  — think {s}", .{@tagName(effort)}) });
                    // The effort is part of the system block: prime it again.
                    primeSession(ui);
                } else {
                    try ui.emit(.{ .notice = try std.fmt.allocPrint(a, "  — {s} is not an effort; known: off, low, medium, high, xhigh", .{name}) });
                }
            },
            .ctx => |size| {
                if (size == 0 or size > config.max_context) {
                    try ui.emit(.{ .notice = try std.fmt.allocPrint(a, "  — context must be between 1 and {d}", .{config.max_context}) });
                } else {
                    // The main loop re-opens the engine: KV capacity is
                    // allocated at open time, so this cannot be done here.
                    ui.ctx_request = size;
                }
            },
            .save => |where| try saveSession(ui, a, root_dir, where),
        },
    }
}

/// `/save`: the session file as markdown. It is derived from the entries, not
/// from the screen, so what is exported is what the model actually saw — and
/// a session that has recorded nothing yet says so rather than writing an
/// empty document.
fn saveSession(ui: *Ui, a: std.mem.Allocator, root_dir: ?[]const u8, where: ?[]const u8) !void {
    const source = ui.log.path orelse {
        try ui.emit(.{ .notice = "  — no session file (no user root): nothing to save" });
        return;
    };
    var diagnostic: session_log.Diagnostic = .{};
    const loaded = session_log.load(a, ui.io, .cwd(), source, &diagnostic) catch |err| {
        const reason = if (diagnostic.line > 0)
            try std.fmt.allocPrint(a, "  — {s} (line {d}) reading {s}", .{ @errorName(err), diagnostic.line, source })
        else if (err == error.FileNotFound)
            "  — nothing recorded in this session yet"
        else
            try std.fmt.allocPrint(a, "  — {s} reading {s}", .{ @errorName(err), source });
        try ui.emit(.{ .notice = reason });
        return;
    };
    defer loaded.deinit();
    const target = if (where) |path|
        try a.dupe(u8, path)
    else if (root_dir) |root|
        try session_log.exportPath(a, root, loaded.header.id, loaded.header.time)
    else {
        try ui.emit(.{ .notice = "  — no user root: give /save a path" });
        return;
    };
    var document: std.Io.Writer.Allocating = .init(a);
    try session_log.exportMarkdown(loaded, &document.writer);
    if (std.fs.path.dirname(target)) |parent| try std.Io.Dir.cwd().createDirPath(ui.io, parent);
    std.Io.Dir.cwd().writeFile(ui.io, .{ .sub_path = target, .data = document.written() }) catch |err| {
        try ui.emit(.{ .notice = try std.fmt.allocPrint(a, "  — {s} writing {s}", .{ @errorName(err), target }) });
        return;
    };
    try ui.emit(.{ .notice = try std.fmt.allocPrint(a, "  — saved {s}", .{target}) });
}

/// `/resume` and `--resume`: replay a saved conversation into a fresh session
/// and continue it. The engine is reset first — a resume is a replay through
/// the profile, never a state restore — and the fresh session file is seeded
/// with the saved entries, so the resumed conversation is complete on disk and
/// can itself be resumed.
fn performResume(ui: *Ui, alloc: std.mem.Allocator, io: std.Io, path: []const u8, root_dir: ?[]const u8, cwd: []const u8, model_path: []const u8, digest: ?[]const u8, log: *session_log.Session) !void {
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const a = arena.allocator();
    var diagnostic: session_log.Diagnostic = .{};
    const loaded = session_log.load(a, io, .cwd(), path, &diagnostic) catch |err| {
        const reason = if (diagnostic.line > 0)
            try std.fmt.allocPrint(a, "  — {s} (line {d}) loading {s}", .{ @errorName(err), diagnostic.line, path })
        else
            try std.fmt.allocPrint(a, "  — {s} loading {s}", .{ @errorName(err), path });
        try ui.emit(.{ .notice = reason });
        return;
    };
    const effort = std.meta.stringToEnum(Profile.Effort, loaded.header.effort) orelse ui.effort;
    var fresh = newSessionLog(alloc, io, root_dir, cwd, model_path, digest, effort, ui.eng.model.session().capacity) catch |err| {
        var note: [96]u8 = undefined;
        try ui.emit(.{ .notice = std.fmt.bufPrint(&note, "  — {s} opening a session for the replay", .{@errorName(err)}) catch "  — could not open a session" });
        return;
    };
    for (loaded.records) |record| {
        fresh.append(record.entry, if (record.time.len > 0) record.time else loaded.header.time) catch break;
    }
    log.deinit();
    log.* = fresh;
    ui.log_failed = false;
    // Forget the running session: the resumed conversation is rendered from
    // the stored entries and prefilled on the next turn.
    ui.eng.model.reset();
    ui.tokens_seen.reset();
    ui.completer.reset();
    ui.agent.resetConversation();
    ui.effort = effort;
    const built = try resume_mod.messages(a, loaded);
    try ui.agent.restore(built);
    ui.tr.reset();
    const rows = try resume_mod.replay(a, ui.tr, loaded, .{ .width = ui.columnsFor(), .th = ui.th });
    if (rows.len > 0) try ui.scr.insertAbove(rows);
    ui.bar = .{};
    ui.status = "resumed";
}

/// Names a new session file for this working directory. A session without a
/// user root (no `HOME`, no `NUCLIS_HOME`) still runs; it just records
/// nothing, which is also what print mode will want.
fn newSessionLog(alloc: std.mem.Allocator, io: std.Io, root_dir: ?[]const u8, cwd: []const u8, model_path: []const u8, digest: ?[]const u8, effort: Profile.Effort, ctx_size: usize) !session_log.Session {
    var id: [32]u8 = undefined;
    var stamp: [20]u8 = undefined;
    return session_log.create(alloc, io, .cwd(), .{
        .root_dir = root_dir,
        .id = session_log.newId(io, &id),
        .time = model.rfc3339(&stamp, std.Io.Timestamp.now(io, .real).toSeconds()),
        .cwd = cwd,
        .model_path = model_path,
        .model_sha256 = digest,
        .effort = @tagName(effort),
        .ctx_size = ctx_size,
    });
}

/// `settings` is the configuration resolved for the agent (defaults < file
/// < flags): the context window, the per-turn output ceiling (a stop token
/// almost always ends the turn earlier; the 2,048 default keeps long code
/// answers whole), the backend, the starting effort, the sampling
/// overrides, and the initial thinking fold.
pub fn run(alloc: std.mem.Allocator, io: std.Io, environ: *const std.process.Environ.Map, model_path: []const u8, root_dir: ?[]const u8, settings: config.Resolved, seed: ?u64, resume_id: ?[]const u8, out: *std.Io.Writer) !void {
    const capacity = settings.ctx_size;
    const limit = settings.max_tokens;
    if (capacity == 0 or capacity > config.max_context or limit == 0 or limit > config.max_output_tokens) return error.InvalidGenerationBudget;
    const backend = settings.backend;
    const kv = settings.kv_precision;
    // Validates the overrides against the starting effort's profile before
    // the model loads; later efforts are rebuilt per turn (see `runTurn`).
    var sampler = try inference.sampling.Sampler.init(seed orelse 0, settings.samplingOptions());
    try out.writeAll("Loading model…\n");
    try out.flush();
    var eng = try engine.Engine.open(alloc, io, model_path, backend, capacity, kv, settings.forced_profile, .none);
    defer eng.deinit();
    // The file's own profile from here on: the configuration guessed one
    // from the catalogue name without opening the file (`config show`).
    const profile = eng.profile orelse return error.UnsupportedPromptTemplate;
    try sampler.setOptions(profile.samplingOptions(settings.think, settings.sampling));
    const logits = try alloc.alloc(f32, eng.vocab.tokens.len);
    defer alloc.free(logits);
    // Sized for sampling regardless of the current temperature: Ctrl-T can
    // switch profiles mid-session.
    const candidates = try alloc.alloc(inference.sampling.Candidate, eng.vocab.tokens.len);
    defer alloc.free(candidates);
    const generated = try alloc.alloc(u32, limit);
    defer alloc.free(generated);
    var history = try inference.sampling.History.init(alloc, eng.vocab.tokens.len);
    defer history.deinit();

    // The prompt editor, seeded from `~/.nuclis/agent/history.jsonl` when
    // there is a user root. A missing or unreadable file is an empty
    // history, never a startup failure.
    var ed: editor.Editor = .{ .alloc = alloc };
    defer ed.deinit();
    const history_path: ?[]u8 = if (root_dir) |dir| try paths.agentPath(alloc, dir, paths.history_file) else null;
    defer if (history_path) |path| alloc.free(path);
    if (history_path) |path| {
        if (prompt_history.load(alloc, io, std.Io.Dir.cwd(), path)) |loaded| {
            defer loaded.deinit();
            try ed.seedHistory(loaded.items);
        } else |_| {}
    }

    // The session file: named now, written at the first turn. Its header
    // records what the conversation ran against — the working directory, the
    // model and the digest its sidecar verified, the context window, the
    // effort — so a file read a year later says what produced it.
    // `realPathFileAlloc` returns a sentinel-terminated slice; the fallback
    // must too, or the coerced `[]u8` frees one byte short of the allocation.
    const cwd = std.Io.Dir.cwd().realPathFileAlloc(io, ".", alloc) catch try alloc.dupeZ(u8, ".");
    defer alloc.free(cwd);
    const digest: ?[]const u8 = blk: {
        // The sidecar's strings live in the allocator it is read with; only
        // the digest outlives this block.
        var arena = std.heap.ArenaAllocator.init(alloc);
        defer arena.deinit();
        const a = arena.allocator();
        const sidecar_path = model.sidecarPath(a, model_path) catch break :blk null;
        const sidecar = model.readSidecar(a, io, .cwd(), sidecar_path) catch break :blk null;
        break :blk if (sidecar) |record| try alloc.dupe(u8, record.sha256) else null;
    };
    defer if (digest) |d| alloc.free(d);
    var log = try newSessionLog(alloc, io, root_dir, cwd, model_path, digest, settings.think, capacity);
    defer log.deinit();

    var term = try terminal.Terminal.init(io, out);
    const th = theme.Theme.detectNamed(environ, settings.theme);
    var scr: screen.Screen = .{ .out = out, .th = th, .caps = screen.Caps.detect(environ), .size = term.size() };
    var tr: transcript.Transcript = .{ .alloc = alloc, .expanded = !settings.fold_thinking };
    defer tr.deinit();
    var comp: choice.Choice = .{ .alloc = alloc };
    defer comp.deinit();
    // The loop and its completion side. The agent owns the conversation; the
    // completer owns the session's consumed-text bookkeeping and points at the
    // engine the main loop may re-open. The driver only presents and feeds
    // input.
    var completer: loop.Completer = .{
        .alloc = alloc,
        .eng = &eng,
        .effort = settings.think,
        .sampler = &sampler,
        .history = &history,
        .buffers = .{ .logits = logits, .candidates = candidates, .generated = generated, .effort = settings.think },
    };
    defer completer.deinit();
    var workspace: tools.Workspace = .{ .io = io, .dir = .cwd(), .root = cwd, .environ = environ };
    var agent: loop.Agent = undefined;
    var ui: Ui = .{ .alloc = alloc, .io = io, .eng = &eng, .term = &term, .scr = &scr, .ed = &ed, .tr = &tr, .log = &log, .comp = &comp, .th = th, .agent = &agent, .completer = &completer, .effort = settings.think, .overrides = settings.sampling, .profile = profile, .tokens_seen = &history, .turn_started = std.Io.Clock.awake.now(io), .notify = terminal.notificationsEnabled(environ) };
    defer ui.deinit();
    // A polling tool reaches back into the driver while it runs, so keys are
    // read and the running call's spinner advances during a long `bash`.
    workspace.tick = .{ .context = &ui, .call = toolTick };
    // The observer carries both cancellation (`interrupt.check`, `Ctrl-C`) and
    // the progress beats the bar animates from.
    var trace: generate.Trace = .{
        .io = io,
        .directory = null,
        .started = std.Io.Clock.awake.now(io),
        .progress = .{ .context = &ui, .call = Ui.onProgress },
    };
    completer.observer = trace.observer();
    agent = try loop.Agent.init(alloc, io, workspace, completer.model(), ui.events(), loop.budget_default);
    agent.result_budget = loop.resultBudget(capacity);
    defer agent.deinit();
    {
        defer term.deinit();
        // Exit leaves the transcript and nothing else: the live region is
        // erased before the lease restores the terminal (defers run in
        // reverse, so this happens first).
        defer scr.finish() catch {};
        // Full-height layout: clear the visible screen (scrollback is kept),
        // put the header at the top, then walk down so the live region
        // anchors at the bottom — which is also what lets insertion above it
        // scroll everything in between.
        try scr.clear();
        {
            // The welcome: the wordmark when it fits, and what this session
            // runs with. Transcript, like everything above the region.
            var arena = std.heap.ArenaAllocator.init(alloc);
            defer arena.deinit();
            const a = arena.allocator();
            const named = settings.entry != null or catalog.find(settings.model) != null;
            ui.model_label = try alloc.dupe(u8, if (named) settings.model else eng.name);
            const welcome = try tui.banner.rows(a, .{
                .version = version,
                .name = if (named) settings.model else null,
                .model = eng.name,
                .backend = @tagName(eng.backend),
                .profile = @tagName(profile),
                .forced = eng.profile_forced,
                .ctx_size = capacity,
                .effort = @tagName(settings.think),
                .workspace = try tui.banner.shortened(a, cwd, environ.get("HOME")),
            }, term.size().columns);
            try scr.insertAbove(welcome);
        }
        try scr.anchor(min_editor_rows + 3);
        primeSession(&ui);
        // A forced profile renders the pinned protocol onto a file whose own
        // template says something else: worth one line, every time.
        if (eng.profile_forced) {
            var note: [160]u8 = undefined;
            ui.emit(.{ .notice = std.fmt.bufPrint(&note, "  — prompt profile {s} forced: the file's chat template is not the pinned one", .{@tagName(profile)}) catch "  — prompt profile forced" }) catch {};
        }
        // `--resume <id>`: locate the session and replay it before the first
        // prompt. A missing id is a notice, not a startup failure.
        if (resume_id) |id| {
            // A path needs no root; an id is resolved through the user root.
            const found = resume_mod.find(alloc, io, root_dir orelse "", cwd, id) catch null;
            if (found) |path| {
                defer alloc.free(path);
                performResume(&ui, alloc, io, path, root_dir, cwd, model_path, digest, &log) catch |err| {
                    ui.status = @errorName(err);
                };
            } else {
                var note: [128]u8 = undefined;
                ui.emit(.{ .notice = std.fmt.bufPrint(&note, "  — no session {s} for this workspace", .{id}) catch "  — no such session" }) catch {};
            }
        }
        while (!ui.quit) {
            try ui.draw();
            try ui.poll(100);
            if (ui.ctx_request) |newcap| {
                ui.ctx_request = null;
                const old_capacity = eng.model.session().capacity;
                ui.status = "resizing context…";
                try ui.draw();
                eng.deinit();
                if (engine.Engine.open(alloc, io, model_path, backend, newcap, kv, settings.forced_profile, .none)) |opened| {
                    eng = opened;
                    ui.eng = &eng;
                    ui.completer.reset();
                    ui.tokens_seen.reset(); // same artifact, same vocabulary; a fresh session
                    ui.agent.result_budget = loop.resultBudget(newcap);
                    ui.record(.{ .context = .{ .ctx_size = newcap } });
                    // The old snapshot belongs to the old engine; prime anew.
                    ui.completer.dropPrimed();
                    primeSession(&ui);
                    ui.status = "ctx resized";
                } else |err| {
                    // Fall back to the previous size rather than lose the
                    // session over an allocation failure.
                    eng = try engine.Engine.open(alloc, io, model_path, backend, old_capacity, kv, settings.forced_profile, .none);
                    ui.eng = &eng;
                    ui.status = @errorName(err);
                }
                continue;
            }
            if (ui.new_session) {
                ui.new_session = false;
                var fresh = newSessionLog(alloc, io, root_dir, cwd, model_path, digest, ui.effort, eng.model.session().capacity) catch null;
                if (fresh) |*value| {
                    log.deinit();
                    log = value.*;
                    ui.log = &log;
                    ui.log_failed = false;
                }
                continue;
            }
            // A `/resume` choice: replay the chosen session now that the
            // picker has named a path, then carry on in it.
            if (ui.resume_request) |path| {
                ui.resume_request = null;
                defer alloc.free(path);
                performResume(&ui, alloc, io, path, root_dir, cwd, model_path, digest, &log) catch |err| {
                    ui.status = @errorName(err);
                    var note: [96]u8 = undefined;
                    ui.emit(.{ .notice = std.fmt.bufPrint(&note, "  — {s} resuming session", .{@errorName(err)}) catch "  — resume failed" }) catch {};
                };
                continue;
            }
            // A message typed while the last turn ran goes out first, without
            // a keystroke: that is what queuing it meant.
            if (ui.queued.items.len > 0 and !ui.submit) {
                try ed.insert(ui.queued.items);
                ui.queued.clearRetainingCapacity();
                ui.submit = true;
            }
            if (!ui.submit) continue;
            ui.submit = false;
            if (ed.isEmpty()) continue;
            if (!std.unicode.utf8ValidateSlice(ed.text())) {
                ui.status = "invalid UTF-8 input";
                continue;
            }
            // The prompt outlives the editor's buffer: recall rewrites it.
            const user = try alloc.dupe(u8, ed.text());
            defer alloc.free(user);
            ed.clear();
            try ed.remember(user);
            // A slash command is an instruction to the agent, not a turn: it
            // never reaches the model, and the loop starts over.
            if (commands.parse(user)) |parsed| {
                try runCommand(&ui, parsed, root_dir, cwd);
                continue;
            }
            if (history_path) |path| {
                var stamp: [20]u8 = undefined;
                const now = model.rfc3339(&stamp, std.Io.Timestamp.now(io, .real).toSeconds());
                // A history write is a convenience, never a reason to lose a turn.
                prompt_history.append(io, std.Io.Dir.cwd(), path, user, now) catch {};
            }
            runTurn(&ui, &sampler, user) catch |err| {
                ui.agent.abortTurn();
                ui.eng.model.reset();
                ui.tokens_seen.reset();
                ui.completer.reset();
                interrupt.clear();
                // The prompt is already in the transcript; say why nothing
                // followed it, and close the turn so the next one starts on
                // its own line. A full window says how full, and what to do.
                var note: [160]u8 = undefined;
                const text = if (err == error.ContextFull) blk: {
                    ui.status = "context full";
                    const overflow = ui.completer.overflow orelse loop.Overflow{ .needed = 0, .capacity = ui.eng.model.session().capacity };
                    break :blk std.fmt.bufPrint(&note, "  — context window full: the step needed {d} tokens (prompt plus output budget) of {d}; raise it with /ctx <n> (or --ctx-size), or start over with /new", .{ overflow.needed, overflow.capacity }) catch "  — context window full";
                } else blk: {
                    ui.status = @errorName(err);
                    break :blk std.fmt.bufPrint(&note, "  — {s}", .{@errorName(err)}) catch "  — failed";
                };
                ui.emit(.{ .notice = text }) catch {};
                ui.emit(.{ .turn_end = .{ .stop = .failure } }) catch {};
            };
        }
    }
}

test "increment is the unseen suffix, or null when the conversation must replay" {
    try std.testing.expectEqualStrings("<|im_end|>\n<|im_start|>user\nmore", loop.increment("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nanswer", "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nanswer<|im_end|>\n<|im_start|>user\nmore").?);
    try std.testing.expectEqualStrings("whole", loop.increment("", "whole").?);
    try std.testing.expect(loop.increment("system A", "system B") == null);
}

test {
    _ = tui;
    _ = prompt_history;
    _ = session_log;
    _ = commands;
    _ = resume_mod;
    _ = print_mode;
    _ = tools;
    _ = loop;
    _ = stream;
}

test "rule fills the requested cell width in either glyph set" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const th: theme.Theme = .{ .kind = .plain };
    for ([_]usize{ 1, 2, 10, 80, 199, 400 }) |cells_w| {
        const r = try Ui.rule(arena.allocator(), cells_w, th);
        try std.testing.expectEqual(3 * cells_w, r.len);
        try std.testing.expectEqualStrings("─", r[0..3]);
        try std.testing.expectEqualStrings("─", r[r.len - 3 ..]);
        const ascii = try Ui.rule(arena.allocator(), cells_w, .{ .kind = .plain, .glyph_set = .ascii });
        try std.testing.expectEqual(cells_w, ascii.len);
    }
}
