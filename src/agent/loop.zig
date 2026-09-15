//! The agent loop: a turn is a sequence of steps, and a step is one completion
//! request followed by the sequential execution of every tool call in the
//! response (docs/agent-spec.md § Agent loop).
//!
//! ```
//! user message ──▶ render + complete ──▶ decoded calls
//!                        ▲                    │
//!                        └── tool results ◀───┘  (until no calls, budget, or cancel)
//! ```
//!
//! The loop owns the application's side of the conversation: the system
//! instructions, the message history, the step budget, and the correlation ids
//! that match a result back to its call. It owns no model syntax — the profile
//! renders the tool definitions and native history and decodes generated calls
//! into typed `events.ToolCall`s — no terminal (the `Events` sink does), and no
//! I/O beyond what the injected `Model` performs. Each call the profile
//! decodes becomes a host id, an assistant history message carrying the
//! `tool_calls`, and one `.tool` result message per call.
//!
//! Two seams make this testable without a model or a terminal:
//!
//! - `Model` is the completion side. `Completer` implements it over
//!   `inference.engine.complete`, keeping the incremental-prefill bookkeeping;
//!   a test supplies a stub that answers from a script.
//! - `Events` is the presentation/session side. The loop sends terminal events
//!   for display and `Record`s for the session file.
const std = @import("std");
const inference = @import("inference");
const engine = @import("../engine.zig");
const tui = @import("../tui/root.zig");
const tools = @import("tools/root.zig");
const stream = @import("stream.zig");

const Allocator = std.mem.Allocator;
const Profile = inference.profiles;
const Event = tui.event.Event;

/// Model completions allowed in one turn before the loop stops. A host
/// constant, never model-supplied (agent-spec § Design principles).
pub const budget_default: usize = 16;

/// User turns kept in one conversation, mirroring the surface's own limit.
pub const max_turns: usize = 256;

/// Where a finished turn stopped.
pub const Stop = enum { done, budget, cancelled };

/// One terminal-event and session-record sink. `send` is the transcript;
/// `record` is the append-only session file, kept separate because an entry is
/// not an event (one turn records one assistant entry and many deltas).
pub const Events = struct {
    context: *anyopaque,
    send: *const fn (*anyopaque, Event) anyerror!void,
    record: *const fn (*anyopaque, Record) anyerror!void,
};

/// One session entry, in the shape the drivers map onto `session.Entry`.
pub const Record = union(enum) {
    user: []const u8,
    assistant: Step,
    tool_result: Result,
};

/// One completed assistant step. All slices borrow only for the call.
pub const Step = struct {
    thinking: []const u8,
    answer: []const u8,
    /// The calls the profile decoded, with their host correlation ids. Empty
    /// on a cancelled step: recording calls without the results that answer
    /// them would produce a session that cannot be loaded again.
    calls: []const Profile.ToolCall = &.{},
    stop: engine.StopReason,
    prompt_tokens: usize = 0,
    generated: usize = 0,
    prefill_seconds: f64 = 0,
    decode_seconds: f64 = 0,
    /// The session had to be replayed for this step (compaction or a reset).
    replayed: bool = false,
};

/// One executed tool call and its typed result.
pub const Result = struct {
    call: u32,
    name: []const u8,
    text: []const u8,
    truncated: bool = false,
    is_error: bool = false,
    /// The transcript's detail row, kept so a resumed session shows it.
    summary: []const u8 = "",
};

/// What one completion step reported. `outcome` is the engine's; `replayed`
/// says whether the conversation had to be rendered from an empty session.
pub const Reply = struct {
    outcome: inference.engine.Outcome,
    replayed: bool = false,
};

/// The completion side of a step. `messages` is the conversation to render
/// (system first) and `tools` are the definitions the profile renders into
/// that system message; the implementation forwards every engine event to
/// `sink`.
pub const Model = struct {
    context: *anyopaque,
    run: *const fn (*anyopaque, messages: []const Profile.Message, definitions: []const Profile.ToolDefinition, sink: *stream.Sink) anyerror!Reply,
    /// Tokens `text` costs in the model's vocabulary; what the result budget
    /// is measured in.
    count: *const fn (*anyopaque, text: []const u8) anyerror!usize,
};

/// Tokens one tool result may occupy: a fixed share of the context window,
/// never below `min_result_budget`. The tools' own byte and line limits stay
/// as absolute ceilings; this is the bound that scales with the window.
pub fn resultBudget(capacity: usize) usize {
    return @max(capacity / 8, min_result_budget);
}
pub const min_result_budget: usize = 256;

/// What did not fit when a completion reported `ContextFull`: the tokens the
/// step needed against the window. For the message the user reads.
pub const Overflow = struct { needed: usize, capacity: usize };

/// One history message, owned by the loop. `reasoning` is empty except on an
/// assistant step. `tool_calls` are the calls the assistant requested (with
/// host ids assigned), and `tool_call_id` marks a `.tool` result.
const Item = struct {
    role: Profile.Role,
    content: []u8,
    reasoning: []u8,
    tool_calls: []Profile.ToolCall = &.{},
    tool_call_id: ?u32 = null,
};

pub const Agent = struct {
    alloc: Allocator,
    io: std.Io,
    workspace: tools.Workspace,
    model: Model,
    events: Events,
    /// The leading system message: identity and workspace. Built once, owned.
    /// The profile renders the tool definitions into this block; the loop
    /// never spells the wire format.
    system: []u8,
    /// The definitions handed to the profile, in registry order. Built once;
    /// the names, descriptions, and schemas borrow the static registry.
    tool_defs: []Profile.ToolDefinition,
    history: std.ArrayList(Item) = .empty,
    budget: usize,
    /// Tokens one tool result may occupy before the loop cuts it
    /// (`resultBudget`); the caller sets it from the window it opened.
    result_budget: usize = resultBudget(8192),
    next_id: u32 = 1,
    /// Model completions completed in the turn in progress; the status bar
    /// shows it against `budget`. Reset at the start of every turn.
    steps_done: usize = 0,
    max_turns: usize = max_turns,
    /// Where the turn in progress begins in `history`; only earlier items may
    /// be dropped by compaction.
    turn_start: usize = 0,

    thinking: std.Io.Writer.Allocating,
    /// The answer content of the step, calls excluded: what the user read and
    /// what the model consumed next time (the calls are rendered separately).
    answer: std.Io.Writer.Allocating,
    /// Calls the profile decoded during this step, in emission order. Owned;
    /// cleared at the start of each step.
    calls: std.ArrayList(Profile.ToolCall) = .empty,
    thinking_open: bool = false,
    thinking_ended: bool = false,
    turn_started: std.Io.Timestamp,

    agg_prompt_tokens: usize = 0,
    agg_generated: usize = 0,
    agg_prefill_seconds: f64 = 0,
    agg_decode_seconds: f64 = 0,
    agg_replayed: bool = false,

    /// Reusable `Profile.Message` scratch for one step.
    messages: std.ArrayList(Profile.Message) = .empty,

    pub fn init(alloc: Allocator, io: std.Io, workspace: tools.Workspace, model: Model, events: Events, budget: usize) !Agent {
        var agent: Agent = .{
            .alloc = alloc,
            .io = io,
            .workspace = workspace,
            .model = model,
            .events = events,
            .system = undefined,
            .tool_defs = undefined,
            .budget = budget,
            .thinking = .init(alloc),
            .answer = .init(alloc),
            .turn_started = std.Io.Clock.awake.now(io),
        };
        errdefer agent.thinking.deinit();
        errdefer agent.answer.deinit();
        agent.system = try systemPrompt(alloc, workspace.root);
        errdefer alloc.free(agent.system);
        agent.tool_defs = try toolDefs(alloc);
        return agent;
    }

    pub fn deinit(self: *Agent) void {
        self.alloc.free(self.system);
        self.alloc.free(self.tool_defs);
        self.clearHistory();
        self.history.deinit(self.alloc);
        self.messages.deinit(self.alloc);
        self.clearCalls();
        self.calls.deinit(self.alloc);
        self.thinking.deinit();
        self.answer.deinit();
        self.* = undefined;
    }

    /// Forgets the conversation but keeps the engine usable as it is: the next
    /// turn replays from the fresh session the model already reset.
    pub fn resetConversation(self: *Agent) void {
        self.clearHistory();
        self.turn_start = 0;
        self.clearCalls();
    }

    /// Rebuilds the conversation from a stored one (`/resume`, `--resume`), so
    /// a resumed session continues instead of starting over. Messages are
    /// appended in order and the host correlation ids continue past the
    /// highest restored one. The caller has already reset the engine session:
    /// this is a replay, never a state restore.
    pub fn restore(self: *Agent, messages: []const Profile.Message) !void {
        var highest: u32 = 0;
        for (messages) |message| {
            try self.appendItem(message.role, message.content, message.reasoning_content, message.tool_calls, message.tool_call_id);
            for (message.tool_calls) |call| highest = @max(highest, call.id);
            if (message.tool_call_id) |id| highest = @max(highest, id);
        }
        self.next_id = @max(self.next_id, highest + 1);
        self.turn_start = self.history.items.len;
    }

    /// Drops the turn in progress after a failure, leaving the earlier
    /// conversation intact. The driver has already shown the prompt; the model
    /// never saw a completed answer, so nothing of it is kept.
    pub fn abortTurn(self: *Agent) void {
        for (self.history.items[self.turn_start..]) |item| self.freeItem(item);
        self.history.shrinkRetainingCapacity(self.turn_start);
    }

    /// Runs one user turn to a stop. Emits the user event and every step's
    /// events; the caller only has to present them and handle the stop.
    pub fn turn(self: *Agent, user: []const u8) !Stop {
        self.turn_start = self.history.items.len;
        if (self.turnCount() >= self.max_turns) return error.ConversationLimit;
        try self.appendItem(.user, user, "", &.{}, null);
        try self.events.send(self.events.context, .{ .user = user });
        try self.events.record(self.events.context, .{ .user = user });

        self.turn_started = std.Io.Clock.awake.now(self.io);
        self.steps_done = 0;
        self.agg_prompt_tokens = 0;
        self.agg_generated = 0;
        self.agg_prefill_seconds = 0;
        self.agg_decode_seconds = 0;
        self.agg_replayed = false;

        const stop = try self.steps();
        try self.emitTurnEnd(stop);
        return stop;
    }

    fn steps(self: *Agent) !Stop {
        var completed: usize = 0;
        while (true) {
            if (completed >= self.budget) {
                try self.events.send(self.events.context, .{ .notice = "  — step budget reached" });
                return .budget;
            }
            const reply = try self.runStep();
            completed += 1;
            self.steps_done = completed;
            self.agg_prompt_tokens += reply.outcome.timing.prompt_tokens;
            self.agg_generated += reply.outcome.timing.generated_tokens;
            self.agg_prefill_seconds += seconds(reply.outcome.timing.prefill);
            self.agg_decode_seconds += seconds(reply.outcome.timing.decode);
            self.agg_replayed = self.agg_replayed or reply.replayed;

            // A cancelled step is display-only: the model never saw its end,
            // so it is left out of what the next step renders — and its calls
            // are not recorded, because a session cannot load an assistant
            // tool call whose result never followed.
            const cancelled = reply.outcome.stop == .cancelled;
            // The assistant turn is the answer content plus the calls the
            // profile decoded; assign their host ids first, so the session
            // records the same correlation ids the history carries.
            const calls = self.calls.items;
            if (!cancelled) for (calls) |*call| {
                call.id = self.next_id;
                self.next_id += 1;
            };
            try self.events.record(self.events.context, .{ .assistant = .{
                .thinking = self.thinking.written(),
                .answer = self.answer.written(),
                .calls = if (cancelled) &.{} else calls,
                .stop = reply.outcome.stop,
                .prompt_tokens = reply.outcome.timing.prompt_tokens,
                .generated = reply.outcome.timing.generated_tokens,
                .prefill_seconds = seconds(reply.outcome.timing.prefill),
                .decode_seconds = seconds(reply.outcome.timing.decode),
                .replayed = reply.replayed,
            } });
            if (cancelled) return .cancelled;
            try self.appendItem(.assistant, self.answer.written(), self.thinking.written(), calls, null);
            if (calls.len == 0) return .done;
            try self.execute(calls);
        }
    }

    /// One completion step, compacting and retrying when the conversation no
    /// longer fits. `ContextFull` with nothing left to drop is the caller's.
    fn runStep(self: *Agent) !Reply {
        while (true) {
            self.beginStep();
            const messages = try self.buildMessages();
            var sink = self.engineSink();
            const reply = self.model.run(self.model.context, messages, self.tool_defs, &sink) catch |err| switch (err) {
                error.ContextFull => {
                    if (!self.dropOldestTurn()) return err;
                    try self.events.send(self.events.context, .{ .notice = "  — older turns dropped from context to fit" });
                    continue;
                },
                else => return err,
            };
            return reply;
        }
    }

    fn beginStep(self: *Agent) void {
        self.clearCalls();
        self.thinking.clearRetainingCapacity();
        self.answer.clearRetainingCapacity();
        self.thinking_open = false;
        self.thinking_ended = false;
    }

    fn buildMessages(self: *Agent) ![]const Profile.Message {
        self.messages.clearRetainingCapacity();
        try self.messages.append(self.alloc, .{ .role = .system, .content = self.system });
        for (self.history.items) |item| {
            try self.messages.append(self.alloc, .{
                .role = item.role,
                .content = item.content,
                .reasoning_content = item.reasoning,
                .tool_calls = item.tool_calls,
                .tool_call_id = item.tool_call_id,
            });
        }
        return self.messages.items;
    }

    /// Runs every call of one response in order, forwards each as a
    /// `tool_call`/`tool_result` pair (with a `diff` in between for a
    /// mutation), records it, and appends each result as a `.tool` history
    /// message. The results the model and the session read carry the unified
    /// diff for a mutation, so both see exactly what changed.
    fn execute(self: *Agent, calls: []const Profile.ToolCall) !void {
        for (calls) |call| {
            const id = call.id;
            // The model names the tools; the transcript shows a humanized row.
            var described = try tools.describe(self.alloc, call.name, call.arguments);
            defer described.deinit(self.alloc);
            try self.events.send(self.events.context, .{ .tool_call = .{ .id = id, .name = call.name, .summary = described.summary, .detail = described.detail } });

            var result = try self.runTool(call);
            defer result.deinit(self.alloc);
            var model_text: []const u8 = result.text;
            var joined: ?[]u8 = null;
            defer if (joined) |bytes| self.alloc.free(bytes);
            if (result.change) |*change| {
                try self.events.send(self.events.context, .{ .diff = .{ .path = change.path, .rows = change.diff.rows } });
                joined = try std.fmt.allocPrint(self.alloc, "{s}\n{s}", .{ result.text, change.diff.unified });
                model_text = joined.?;
            }

            // The tool's own bounds are host ceilings; this cut is the one
            // that scales with the window, so one result cannot fill it.
            var fitted = try self.fit(call, model_text);
            defer if (fitted) |*f| f.deinit(self.alloc);
            var truncated = result.truncated;
            var summary: []const u8 = result.summary orelse "";
            var cut_summary: ?[]u8 = null;
            defer if (cut_summary) |s| self.alloc.free(s);
            if (fitted) |f| {
                model_text = f.text;
                truncated = true;
                cut_summary = if (summary.len > 0)
                    try std.fmt.allocPrint(self.alloc, "{s} · cut to {d} lines for the context", .{ summary, f.kept_lines })
                else
                    try std.fmt.allocPrint(self.alloc, "cut to {d} of {d} lines for the context", .{ f.kept_lines, f.total_lines });
                summary = cut_summary.?;
            }

            try self.events.send(self.events.context, .{ .tool_result = .{
                .id = id,
                .text = model_text,
                .truncated = truncated,
                .is_error = result.is_error,
                .summary = summary,
            } });
            try self.events.record(self.events.context, .{ .tool_result = .{
                .call = id,
                .name = call.name,
                .text = model_text,
                .truncated = truncated,
                .is_error = result.is_error,
                .summary = summary,
            } });
            try self.appendItem(.tool, model_text, "", &.{}, id);
        }
    }

    /// A result cut to the budget: the kept prefix plus a note that says what
    /// was left out and how to ask for it.
    const Fitted = struct {
        text: []u8,
        kept_lines: usize,
        total_lines: usize,

        fn deinit(self: *Fitted, alloc: Allocator) void {
            alloc.free(self.text);
            self.* = undefined;
        }
    };

    /// Null when `text` fits `result_budget`; otherwise the longest prefix at a
    /// line boundary that does, found by scaling the byte cut with the
    /// measured token density and recounting, plus the continuation note.
    fn fit(self: *Agent, call: Profile.ToolCall, text: []const u8) !?Fitted {
        var tokens = try self.model.count(self.model.context, text);
        if (tokens <= self.result_budget) return null;
        const total_lines = countLines(text);
        var keep = text.len;
        var rounds: usize = 0;
        while (tokens > self.result_budget and keep > 0) : (rounds += 1) {
            const density = @as(f64, @floatFromInt(keep)) / @as(f64, @floatFromInt(tokens));
            var target: usize = @intFromFloat(@as(f64, @floatFromInt(self.result_budget)) * density * 0.9);
            if (target >= keep) target = keep - 1;
            // Back up to a line boundary, or, when the first line alone is too
            // long (or the estimate keeps missing), to a UTF-8 boundary.
            const newline = std.mem.lastIndexOfScalar(u8, text[0..target], '\n');
            keep = if (newline != null and newline.? > 0 and rounds < 8) newline.? else utf8Boundary(text, target);
            tokens = try self.model.count(self.model.context, text[0..keep]);
        }
        const kept_lines = countLines(text[0..keep]);
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(self.alloc);
        try out.appendSlice(self.alloc, text[0..keep]);
        try out.print(self.alloc, "\n[truncated to fit the context: {d} of {d} lines shown", .{ kept_lines, total_lines });
        if (std.mem.eql(u8, call.name, "read_file")) {
            try out.print(self.alloc, "; continue with read_file offset={d}", .{readOffset(self.alloc, call.arguments) + kept_lines});
        } else {
            try out.appendSlice(self.alloc, "; narrow the request for the rest");
        }
        try out.appendSlice(self.alloc, "]");
        return .{ .text = try out.toOwnedSlice(self.alloc), .kept_lines = kept_lines, .total_lines = total_lines };
    }

    /// Looks up a registered tool and runs it. An unknown name is a typed
    /// error result, not a failure (agent-spec § Design principles).
    fn runTool(self: *Agent, call: Profile.ToolCall) Allocator.Error!tools.Result {
        const tool = tools.find(call.name) orelse
            return tools.fail(self.alloc, "{s} is not a tool I have", .{call.name});
        return tool.run(self.workspace, self.alloc, call.arguments);
    }

    /// Drops the oldest prior turn from the model's view. Returns false when
    /// only the turn in progress remains. A turn boundary is the next user
    /// message, so a turn's calls and results are dropped together and the
    /// history stays valid.
    fn dropOldestTurn(self: *Agent) bool {
        if (self.turn_start == 0) return false;
        var end = self.turn_start;
        var i: usize = 1;
        while (i < self.turn_start) : (i += 1) {
            if (self.history.items[i].role == .user) {
                end = i;
                break;
            }
        }
        for (self.history.items[0..end]) |item| self.freeItem(item);
        const remaining = self.history.items.len - end;
        std.mem.copyForwards(Item, self.history.items[0..remaining], self.history.items[end..]);
        self.history.shrinkRetainingCapacity(remaining);
        self.turn_start -= end;
        return true;
    }

    fn appendItem(self: *Agent, role: Profile.Role, content: []const u8, reasoning: []const u8, tool_calls: []const Profile.ToolCall, tool_call_id: ?u32) !void {
        const owned_content = try self.alloc.dupe(u8, content);
        errdefer self.alloc.free(owned_content);
        const owned_reasoning = try self.alloc.dupe(u8, reasoning);
        errdefer self.alloc.free(owned_reasoning);
        const calls = try self.alloc.alloc(Profile.ToolCall, tool_calls.len);
        var done: usize = 0;
        errdefer {
            for (calls[0..done]) |call| {
                self.alloc.free(call.name);
                self.alloc.free(call.arguments);
            }
            self.alloc.free(calls);
        }
        for (tool_calls, 0..) |call, i| {
            const name = try self.alloc.dupe(u8, call.name);
            errdefer self.alloc.free(name);
            const arguments = try self.alloc.dupe(u8, call.arguments);
            calls[i] = .{ .id = call.id, .name = name, .arguments = arguments };
            done = i + 1;
        }
        try self.history.append(self.alloc, .{
            .role = role,
            .content = owned_content,
            .reasoning = owned_reasoning,
            .tool_calls = calls,
            .tool_call_id = tool_call_id,
        });
    }

    fn clearHistory(self: *Agent) void {
        for (self.history.items) |item| self.freeItem(item);
        self.history.clearRetainingCapacity();
    }

    fn clearCalls(self: *Agent) void {
        for (self.calls.items) |call| {
            self.alloc.free(call.name);
            self.alloc.free(call.arguments);
        }
        self.calls.clearRetainingCapacity();
    }

    fn freeItem(self: *Agent, item: Item) void {
        self.alloc.free(item.content);
        self.alloc.free(item.reasoning);
        for (item.tool_calls) |call| {
            self.alloc.free(call.name);
            self.alloc.free(call.arguments);
        }
        if (item.tool_calls.len != 0) self.alloc.free(item.tool_calls);
    }

    fn turnCount(self: *const Agent) usize {
        var count: usize = 0;
        for (self.history.items) |item| {
            if (item.role == .user) count += 1;
        }
        return count;
    }

    fn engineSink(self: *Agent) stream.Sink {
        return .{ .context = self, .call = engineEvent };
    }

    fn engineEvent(context: *anyopaque, e: inference.events.Event) anyerror!void {
        const self: *Agent = @ptrCast(@alignCast(context));
        try self.feed(e);
    }

    /// Maps one engine event: reasoning and answer text forward live, a
    /// decoded `tool_call` is retained for execution after the step, and
    /// `stop` is the loop's to publish.
    fn feed(self: *Agent, e: inference.events.Event) !void {
        switch (e) {
            .thinking => |text| {
                self.thinking_open = true;
                try self.thinking.writer.writeAll(text);
                try self.events.send(self.events.context, .{ .thinking_delta = text });
            },
            .answer => |text| {
                if (self.thinking_open and !self.thinking_ended) {
                    self.thinking_ended = true;
                    try self.events.send(self.events.context, .{ .thinking_end = seconds(self.turn_started.durationTo(std.Io.Clock.awake.now(self.io))) });
                }
                try self.answer.writer.writeAll(text);
                try self.events.send(self.events.context, .{ .answer_delta = text });
            },
            // The profile decoded a native call; the host assigns its
            // correlation id once the step completes.
            .tool_call => |call| {
                const name = try self.alloc.dupe(u8, call.name);
                errdefer self.alloc.free(name);
                const arguments = try self.alloc.dupe(u8, call.arguments);
                errdefer self.alloc.free(arguments);
                try self.calls.append(self.alloc, .{ .id = 0, .name = name, .arguments = arguments });
            },
            .stop => {},
        }
    }

    fn emitTurnEnd(self: *Agent, stop: Stop) !void {
        try self.events.send(self.events.context, .{ .turn_end = .{
            .stop = switch (stop) {
                .done => .eos,
                .budget => .token_budget,
                .cancelled => .cancelled,
            },
            .stats = .{
                .prompt_tokens = self.agg_prompt_tokens,
                .generated = self.agg_generated,
                .prefill_seconds = self.agg_prefill_seconds,
                .decode_seconds = self.agg_decode_seconds,
                .replayed = self.agg_replayed,
            },
        } });
    }
};

/// The leading system message: identity and workspace. The profile renders
/// the tool definitions into this block itself, so the loop never spells the
/// wire format.
fn systemPrompt(alloc: Allocator, root: []const u8) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(alloc);
    errdefer out.deinit();
    try out.writer.print(
        "You are nuclis, a coding agent working in {s}. " ++
            "Complete the user's task with the available tools. Read a file before you change it, " ++
            "make one change at a time, and say what you did when you are finished. " ++
            "Never invent a tool result: wait for the output before you continue.\n",
        .{root},
    );
    return out.toOwnedSlice();
}

/// The tool definitions handed to the profile, in registry order. The names,
/// descriptions, and schemas borrow the static registry; the slice is owned.
fn toolDefs(alloc: Allocator) Allocator.Error![]Profile.ToolDefinition {
    const defs = try alloc.alloc(Profile.ToolDefinition, tools.all.len);
    for (tools.all, 0..) |tool, i| {
        defs[i] = .{ .name = tool.name, .description = tool.description, .parameters = tool.parameters };
    }
    return defs;
}

/// The text still to prefill for `full` given what the model has consumed, or
/// null when the conversation must be replayed from an empty session.
pub fn increment(seen: []const u8, full: []const u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, full, seen)) return null;
    return full[seen.len..];
}

fn countLines(text: []const u8) usize {
    if (text.len == 0) return 0;
    const trailing: usize = if (text[text.len - 1] == '\n') 1 else 0;
    return std.mem.count(u8, text, "\n") + 1 - trailing;
}

/// `at` moved back to the start of the code point it falls in.
fn utf8Boundary(text: []const u8, at: usize) usize {
    var i = at;
    while (i > 0 and i < text.len and (text[i] & 0xC0) == 0x80) i -= 1;
    return i;
}

/// The 1-based `offset` a `read_file` call asked for, 1 when absent or
/// unreadable: the continuation note counts from it.
fn readOffset(alloc: Allocator, arguments: []const u8) usize {
    const parsed = std.json.parseFromSlice(struct { offset: ?usize = null }, alloc, arguments, .{ .ignore_unknown_fields = true }) catch return 1;
    defer parsed.deinit();
    return parsed.value.offset orelse 1;
}

fn seconds(duration: std.Io.Duration) f64 {
    return @as(f64, @floatFromInt(duration.nanoseconds)) / std.time.ns_per_s;
}

/// The real `Model`: renders through the artifact's profile and completes one
/// step, keeping the session's consumed-text bookkeeping so a growing
/// conversation prefills only its remainder when it can.
pub const Completer = struct {
    alloc: Allocator,
    eng: *engine.Engine,
    effort: Profile.Effort,
    sampler: *inference.sampling.Sampler,
    /// Mirrors the session for the sampler's penalties; reset wherever the
    /// session is.
    history: ?*inference.sampling.History,
    buffers: inference.engine.CompletionBuffers,
    observer: ?inference.observer.Observer = null,
    /// Rendered prompt plus generated text the session has consumed. Empty
    /// after a reset, which is what makes the next render replay.
    seen: std.ArrayList(u8) = .empty,
    /// Set when `run` last reported `ContextFull`, for the caller's message.
    overflow: ?Overflow = null,

    pub fn model(self: *Completer) Model {
        return .{ .context = self, .run = run, .count = count };
    }

    fn count(context: *anyopaque, text: []const u8) anyerror!usize {
        const self: *Completer = @ptrCast(@alignCast(context));
        const tokens = try self.eng.encode(text);
        defer self.alloc.free(tokens);
        return tokens.len;
    }

    /// Forgets what the session consumed, so the next step renders from an
    /// empty session. Used when the engine is re-opened or a new conversation
    /// starts; the session itself is reset by `inference` on the next render.
    pub fn reset(self: *Completer) void {
        self.seen.clearRetainingCapacity();
    }

    pub fn deinit(self: *Completer) void {
        self.seen.deinit(self.alloc);
    }

    fn run(context: *anyopaque, messages: []const Profile.Message, definitions: []const Profile.ToolDefinition, sink: *stream.Sink) anyerror!Reply {
        const self: *Completer = @ptrCast(@alignCast(context));
        var replayed = false;
        const full = try self.eng.render(messages, definitions, self.effort);
        defer self.alloc.free(full);
        const remainder = increment(self.seen.items, full) orelse blk: {
            self.eng.model.reset();
            if (self.history) |h| h.reset();
            self.seen.clearRetainingCapacity();
            replayed = true;
            break :blk full;
        };
        const tokens = try self.eng.encode(remainder);
        defer self.alloc.free(tokens);
        const session = self.eng.model.session();
        if (session.position + tokens.len > session.capacity or
            self.buffers.generated.len > session.capacity - session.position - tokens.len)
        {
            self.overflow = .{ .needed = session.position + tokens.len + self.buffers.generated.len, .capacity = session.capacity };
            return error.ContextFull;
        }
        const outcome = try inference.engine.complete(
            self.eng,
            tokens,
            self.buffers.generated.len,
            self.sampler,
            self.history,
            self.buffers,
            self.observer,
            sink,
        );
        // The model consumed the prompt and every generated token except the
        // last one sampled (the stop token or the budget's final token).
        try self.seen.appendSlice(self.alloc, remainder);
        const timing = outcome.timing;
        const fed = if (timing.generated_tokens > 0) self.buffers.generated[0 .. timing.generated_tokens - 1] else self.buffers.generated[0..0];
        const fed_text = try inference.bpe.decode(self.alloc, &self.eng.vocab, fed, true, .{});
        defer self.alloc.free(fed_text);
        try self.seen.appendSlice(self.alloc, fed_text);
        if (outcome.stop == .cancelled) self.seen.clearRetainingCapacity();
        return .{ .outcome = outcome, .replayed = replayed };
    }
};

// ----- tests -----

const testing = std.testing;

/// A scripted `Model`: each `run` emits the next answer text and, when one is
/// scripted, the calls the profile would have decoded. No engine, no tokens,
/// no clock.
const Stub = struct {
    answers: []const []const u8,
    /// One entry per step that produced calls; missing steps produce none.
    calls: []const []const Profile.ToolCall = &.{},
    stops: []const engine.StopReason = &.{},
    index: usize = 0,
    /// When set, the run at this answer index reports `ContextFull` once
    /// instead of answering, to exercise compaction.
    context_full_at: ?usize = null,

    fn model(self: *Stub) Model {
        return .{ .context = self, .run = run, .count = count };
    }
    /// Four bytes per token: a fixed density the budget tests can compute.
    fn count(_: *anyopaque, text: []const u8) anyerror!usize {
        return (text.len + 3) / 4;
    }
    fn run(context: *anyopaque, messages: []const Profile.Message, definitions: []const Profile.ToolDefinition, sink: *stream.Sink) anyerror!Reply {
        _ = messages;
        _ = definitions;
        const self: *Stub = @ptrCast(@alignCast(context));
        if (self.context_full_at) |at| {
            if (self.index == at) {
                self.context_full_at = null;
                return error.ContextFull;
            }
        }
        if (self.index >= self.answers.len) return error.NoScript;
        const i = self.index;
        self.index += 1;
        const text = self.answers[i];
        // Feed in two pieces: the sink must survive a split piece.
        if (text.len > 0) {
            const half = text.len / 2;
            try sink.send(.{ .answer = text[0..half] });
            try sink.send(.{ .answer = text[half..] });
        }
        if (i < self.calls.len) {
            for (self.calls[i]) |call| try sink.send(.{ .tool_call = call });
        }
        return .{ .outcome = .{
            .stop = if (i < self.stops.len) self.stops[i] else .eos,
            .timing = .{ .prompt_tokens = 1, .generated_tokens = 1 },
        } };
    }
};

const Capture = struct {
    alloc: Allocator,
    records: usize = 0,
    assistant_records: usize = 0,
    /// Calls carried by recorded assistant entries: the session must see the
    /// same host ids the history does.
    recorded_calls: usize = 0,
    answers: std.Io.Writer.Allocating,
    results: std.ArrayList([]u8) = .empty,
    calls: usize = 0,
    diffs: usize = 0,
    turn_end: usize = 0,

    fn init(alloc: Allocator) Capture {
        return .{ .alloc = alloc, .answers = .init(alloc) };
    }
    fn deinit(self: *Capture) void {
        self.answers.deinit();
        for (self.results.items) |text| self.alloc.free(text);
        self.results.deinit(self.alloc);
    }
    fn eventsSeam(self: *Capture) Events {
        return .{ .context = self, .send = send, .record = record };
    }
    fn send(context: *anyopaque, e: Event) anyerror!void {
        const self: *Capture = @ptrCast(@alignCast(context));
        switch (e) {
            .answer_delta => |text| try self.answers.writer.writeAll(text),
            .tool_call => self.calls += 1,
            .tool_result => |result| try self.results.append(self.alloc, try self.alloc.dupe(u8, result.text)),
            .diff => self.diffs += 1,
            .turn_end => self.turn_end += 1,
            else => {},
        }
    }
    fn record(context: *anyopaque, entry: Record) anyerror!void {
        const self: *Capture = @ptrCast(@alignCast(context));
        self.records += 1;
        if (entry == .assistant) {
            self.assistant_records += 1;
            self.recorded_calls += entry.assistant.calls.len;
        }
    }
};

/// A temp workspace plus the canonical root the tests free.
const Fixture = struct {
    tmp: testing.TmpDir,
    ws: tools.Workspace,
    root: [:0]u8,

    fn init(alloc: Allocator) !Fixture {
        var tmp = testing.tmpDir(.{});
        const root = try tmp.dir.realPathFileAlloc(testing.io, ".", alloc);
        return .{ .tmp = tmp, .ws = .{ .io = testing.io, .dir = tmp.dir, .root = root }, .root = root };
    }
    fn deinit(self: *Fixture, alloc: Allocator) void {
        self.tmp.cleanup();
        alloc.free(self.root);
    }
};

const read_hello: Profile.ToolCall = .{ .id = 0, .name = "read_file", .arguments = "{\"path\":\"hello.txt\"}" };
const read_a: Profile.ToolCall = .{ .id = 0, .name = "read_file", .arguments = "{\"path\":\"a.txt\"}" };
const write_new: Profile.ToolCall = .{ .id = 0, .name = "write_file", .arguments = "{\"path\":\"new.txt\",\"content\":\"hello\"}" };

test "one call then an answer: the loop executes the call and sends the result back" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "hello.txt", .data = "hi there" });

    var stub: Stub = .{
        .answers = &.{ "Let me read it.\n", "The file says hi there." },
        .calls = &.{&.{read_hello}},
    };
    var capture = Capture.init(alloc);
    defer capture.deinit();
    var agent = try Agent.init(alloc, testing.io, fixture.ws, stub.model(), capture.eventsSeam(), budget_default);
    defer agent.deinit();

    const stop = try agent.turn("what is in hello.txt?");
    try testing.expectEqual(Stop.done, stop);
    try testing.expectEqual(@as(usize, 1), capture.calls);
    try testing.expectEqual(@as(usize, 1), capture.results.items.len);
    try testing.expectEqualStrings("hi there", capture.results.items[0]);
    // The answer stream is the model's text, calls excluded.
    try testing.expectEqualStrings("Let me read it.\nThe file says hi there.", capture.answers.written());
    try testing.expectEqual(@as(usize, 1), capture.turn_end);
    try testing.expectEqual(@as(usize, 2), capture.assistant_records);
    // The session records the call with the same host id the history carries.
    try testing.expectEqual(@as(usize, 1), capture.recorded_calls);
    // The conversation is native: user, assistant(answer + calls), tool, assistant.
    try testing.expectEqual(@as(usize, 4), agent.history.items.len);
    try testing.expectEqual(Profile.Role.assistant, agent.history.items[1].role);
    try testing.expectEqual(@as(usize, 1), agent.history.items[1].tool_calls.len);
    try testing.expectEqualStrings("read_file", agent.history.items[1].tool_calls[0].name);
    try testing.expectEqualStrings("Let me read it.\n", agent.history.items[1].content);
    try testing.expectEqual(Profile.Role.tool, agent.history.items[2].role);
    try testing.expectEqual(agent.history.items[1].tool_calls[0].id, agent.history.items[2].tool_call_id.?);
    try testing.expect(std.mem.indexOf(u8, agent.history.items[2].content, "hi there") != null);
    try testing.expectEqualStrings("The file says hi there.", agent.history.items[3].content);
}

test "a mutation sends a diff event and the model reads the unified change" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    var stub: Stub = .{
        .answers = &.{ "", "done" },
        .calls = &.{&.{write_new}},
    };
    var capture = Capture.init(alloc);
    defer capture.deinit();
    var agent = try Agent.init(alloc, testing.io, fixture.ws, stub.model(), capture.eventsSeam(), budget_default);
    defer agent.deinit();

    try testing.expectEqual(Stop.done, try agent.turn("create it"));
    try testing.expectEqual(@as(usize, 1), capture.calls);
    try testing.expectEqual(@as(usize, 1), capture.diffs);
    try testing.expectEqual(@as(usize, 1), capture.results.items.len);
    // The model's `.tool` result carries the unified diff, not just the status
    // line, so it can see exactly what it changed.
    try testing.expect(std.mem.indexOf(u8, agent.history.items[2].content, "+hello") != null);
    const contents = try fixture.tmp.dir.readFileAlloc(testing.io, "new.txt", alloc, .limited(1024));
    defer alloc.free(contents);
    try testing.expectEqualStrings("hello", contents);
}

test "a result over the context budget is cut at a line and told how to continue" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    // 100 lines of 10 bytes: 1,000 bytes, 250 stub tokens.
    var content: std.Io.Writer.Allocating = .init(alloc);
    defer content.deinit();
    for (1..101) |i| try content.writer.print("line {d:0>4}\n", .{i});
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "long.txt", .data = content.written() });
    var stub: Stub = .{
        .answers = &.{ "", "ok" },
        .calls = &.{&.{.{ .id = 0, .name = "read_file", .arguments = "{\"path\":\"long.txt\",\"offset\":11}" }}},
    };
    var capture = Capture.init(alloc);
    defer capture.deinit();
    var agent = try Agent.init(alloc, testing.io, fixture.ws, stub.model(), capture.eventsSeam(), budget_default);
    defer agent.deinit();
    agent.result_budget = 50; // 200 bytes: 20 lines of the 90 the read returns

    try testing.expectEqual(Stop.done, try agent.turn("read it"));
    const shown = capture.results.items[0];
    // The kept prefix ends on a line boundary and fits the budget.
    const note_at = std.mem.indexOf(u8, shown, "\n[truncated to fit the context: ") orelse return error.TestUnexpectedResult;
    try testing.expect(Stub.count(undefined, shown[0..note_at]) catch unreachable <= 50);
    try testing.expect(shown[note_at - 1] != '\n');
    try testing.expect(std.mem.startsWith(u8, shown, "line 0011\n"));
    // The note counts from the call's own offset.
    const kept = countLines(shown[0..note_at]);
    var expected: [64]u8 = undefined;
    const hint = try std.fmt.bufPrint(&expected, "{d} of 90 lines shown; continue with read_file offset={d}]", .{ kept, 11 + kept });
    try testing.expect(std.mem.indexOf(u8, shown, hint) != null);
    // The model reads the cut text, not the whole file.
    try testing.expectEqualStrings(shown, agent.history.items[2].content);

    // A result that fits is left alone.
    try testing.expect((try agent.fit(read_a, "short")) == null);
}

test "a single line over the budget is cut inside the line at a code point" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    var stub: Stub = .{ .answers = &.{"x"} };
    var capture = Capture.init(alloc);
    defer capture.deinit();
    var agent = try Agent.init(alloc, testing.io, fixture.ws, stub.model(), capture.eventsSeam(), budget_default);
    defer agent.deinit();
    agent.result_budget = 4;
    // 40 bytes of two-byte code points on one line: 10 stub tokens.
    var fitted = (try agent.fit(read_a, "éééééééééééééééééééé")).?;
    defer fitted.deinit(alloc);
    const note_at = std.mem.indexOf(u8, fitted.text, "\n[truncated").?;
    try testing.expect(note_at > 0 and note_at <= 16);
    try testing.expect(std.unicode.utf8ValidateSlice(fitted.text[0..note_at]));
    try testing.expectEqual(@as(usize, 1), fitted.total_lines);
}

test "the step budget stops a model that keeps calling" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "a" });
    var stub: Stub = .{
        .answers = &.{ "why?\n", "", "", "" },
        .calls = &.{ &.{read_a}, &.{read_a}, &.{read_a}, &.{read_a} },
    };
    var capture = Capture.init(alloc);
    defer capture.deinit();
    var agent = try Agent.init(alloc, testing.io, fixture.ws, stub.model(), capture.eventsSeam(), 2);
    defer agent.deinit();

    try testing.expectEqual(Stop.budget, try agent.turn("go"));
    try testing.expectEqual(@as(usize, 2), capture.calls); // two completions, each one call
    try testing.expectEqual(@as(usize, 2), capture.results.items.len);
    // The status bar reads this against the budget.
    try testing.expectEqual(@as(usize, 2), agent.steps_done);
    try testing.expectEqual(@as(usize, 2), agent.budget);
}

test "a cancelled step stops the loop before any call runs" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "a" });
    var stub: Stub = .{
        .answers = &.{"go"},
        .calls = &.{&.{read_a}},
        .stops = &.{.cancelled},
    };
    var capture = Capture.init(alloc);
    defer capture.deinit();
    var agent = try Agent.init(alloc, testing.io, fixture.ws, stub.model(), capture.eventsSeam(), budget_default);
    defer agent.deinit();

    try testing.expectEqual(Stop.cancelled, try agent.turn("go"));
    try testing.expectEqual(@as(usize, 0), capture.calls);
    try testing.expectEqual(@as(usize, 0), capture.results.items.len);
}

test "an unknown tool is an error result the model reads, not a failure" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    const unknown: Profile.ToolCall = .{ .id = 0, .name = "nope", .arguments = "{\"x\":1}" };
    var stub: Stub = .{
        .answers = &.{ "", "I could not do that." },
        .calls = &.{&.{unknown}},
    };
    var capture = Capture.init(alloc);
    defer capture.deinit();
    var agent = try Agent.init(alloc, testing.io, fixture.ws, stub.model(), capture.eventsSeam(), budget_default);
    defer agent.deinit();

    try testing.expectEqual(Stop.done, try agent.turn("go"));
    try testing.expectEqual(@as(usize, 1), capture.results.items.len);
    try testing.expectEqualStrings("nope is not a tool I have", capture.results.items[0]);
}

test "increment is the unseen suffix, or null when the conversation must replay" {
    try testing.expectEqualStrings("<|im_end|>\n<|im_start|>user\nmore", increment("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nanswer", "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nanswer<|im_end|>\n<|im_start|>user\nmore").?);
    try testing.expectEqualStrings("whole", increment("", "whole").?);
    try testing.expect(increment("system A", "system B") == null);
}

test "restore rebuilds the conversation and continues the correlation ids" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "data" });
    var stub: Stub = .{
        .answers = &.{ "again", "done" },
        .calls = &.{&.{read_a}},
    };
    var capture = Capture.init(alloc);
    defer capture.deinit();
    var agent = try Agent.init(alloc, testing.io, fixture.ws, stub.model(), capture.eventsSeam(), budget_default);
    defer agent.deinit();
    const restored = [_]Profile.Message{
        .{ .role = .user, .content = "first" },
        .{ .role = .assistant, .content = "looking\n", .tool_calls = &.{.{ .id = 5, .name = "read_file", .arguments = "{\"path\":\"a.txt\"}" }} },
        .{ .role = .tool, .content = "data", .tool_call_id = 5 },
    };
    try agent.restore(&restored);
    try testing.expectEqual(@as(usize, 3), agent.history.items.len);
    try testing.expectEqual(@as(u32, 6), agent.next_id);
    // The next turn continues the conversation: the restored items are its
    // prefix, and the host id picks up past the highest restored one.
    try testing.expectEqual(Stop.done, try agent.turn("more"));
    // user, assistant(call), tool, assistant — restored three first.
    try testing.expectEqual(@as(usize, 7), agent.history.items.len);
    try testing.expectEqualStrings("first", agent.history.items[0].content);
    try testing.expectEqual(Profile.Role.assistant, agent.history.items[1].role);
    try testing.expectEqual(@as(u32, 5), agent.history.items[1].tool_calls[0].id);
    try testing.expectEqual(Profile.Role.tool, agent.history.items[2].role);
    // The new call's id follows the restored one.
    try testing.expectEqual(@as(u32, 6), agent.history.items[4].tool_calls[0].id);
}

test "the tools block the profile renders is pinned to its measured size" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const defs = try toolDefs(a);
    const sys = try systemPrompt(a, "/w");
    const messages = [_]Profile.Message{
        .{ .role = .system, .content = sys },
        .{ .role = .user, .content = "read a then run it" },
    };
    const with_tools = try Profile.qwen38.render(a, &messages, defs, .low, .{});
    const without = try Profile.qwen38.render(a, &messages, &.{}, .low, .{});
    // Measured on the pinned Qwen artifact (2026-09-14): the six-tool block
    // added 720 prompt tokens / 2,896 bytes to the system prompt, ~9% of an
    // 8,192-token context before any conversation; `read_file`'s description
    // grew it to 2,959 bytes on 2026-09-15. Pinned so a tool description
    // edit cannot grow the context silently.
    try testing.expectEqual(@as(usize, 2959), with_tools.len - without.len);
}

test "compaction drops a whole prior turn, tool response included" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "a" });
    var stub: Stub = .{
        .answers = &.{ "", "first", "second" },
        .calls = &.{&.{read_a}},
        .context_full_at = 2,
    };
    var capture = Capture.init(alloc);
    defer capture.deinit();
    var agent = try Agent.init(alloc, testing.io, fixture.ws, stub.model(), capture.eventsSeam(), budget_default);
    defer agent.deinit();

    try testing.expectEqual(Stop.done, try agent.turn("one"));
    // Turn one is [user, assistant(call), tool, assistant].
    try testing.expectEqual(@as(usize, 4), agent.history.items.len);
    try testing.expectEqual(Profile.Role.tool, agent.history.items[2].role);

    try testing.expectEqual(Stop.done, try agent.turn("two"));
    // Compaction dropped all of turn one, including its tool result, so the
    // history is exactly the second turn and stays structurally valid.
    try testing.expectEqual(@as(usize, 2), agent.history.items.len);
    try testing.expectEqualStrings("two", agent.history.items[0].content);
    try testing.expectEqual(Profile.Role.user, agent.history.items[0].role);
    try testing.expectEqualStrings("second", agent.history.items[1].content);
}
