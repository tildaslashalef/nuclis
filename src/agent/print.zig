//! `nuclis agent --print`: one turn, no terminal.
//!
//! The same engine, the same profile, the same events — written to a pipe
//! instead of painted into a region. It is the scripting entry point for the
//! agent and the harness its loop is tested through, because a stream of JSON
//! lines is something a test can assert on while a terminal is not.
//!
//! Two forms, both of them the turn's events:
//!
//! - **text** (the default) — the answer as the model wrote it, streamed as
//!   it arrives. Nothing else: this is what the terminal surface *showed*,
//!   whose thinking is folded away and whose markdown is the model's own.
//! - **`--json`** — one JSON object per line, every event including the
//!   reasoning text and the progress beats. A reader that wants the thinking,
//!   the rates, or the stop reason asks for this form.
//!
//! The turn runs through the agent loop (`loop.zig`), so a print turn may
//! execute tools: the model's call, each bounded result, and the final answer
//! all appear in the event stream. A session file is written only when
//! `--session <path>` names one, so a scripted turn leaves nothing behind
//! unless it was asked to (agent-spec § Print mode).
const std = @import("std");
const inference = @import("inference");
const engine = @import("../engine.zig");
const generate = @import("../generate.zig");
const config = @import("../config.zig");
const interrupt = @import("../interrupt.zig");
const model = @import("../model.zig");
const paths = @import("../paths.zig");
const tui = @import("../tui/root.zig");
const session_log = @import("session.zig");
const loop = @import("loop.zig");
const tools = @import("tools/root.zig");
const resume_mod = @import("resume.zig");
const view = tui.view;

const Event = tui.event.Event;

/// What print mode writes to, and how.
const Printer = struct {
    out: *std.Io.Writer,
    json: bool,
    /// Set once the answer has started, so the text form can end the turn
    /// with a newline only when something was written.
    wrote: bool = false,

    pub fn send(self: *Printer, e: Event) !void {
        if (self.json) {
            try e.writeJson(self.out);
            try self.out.flush();
            return;
        }
        switch (e) {
            .answer_delta => |text| {
                // Untrusted model text on a pipe: control bytes are stripped
                // here exactly as the terminal surface strips them.
                try view.safe(self.out, text);
                self.wrote = true;
                try self.out.flush();
            },
            else => {},
        }
    }
};

/// Runs one turn and returns when it is done. Errors are the caller's to
/// report: `main` prints them with the diagnostic, as for every other command.
pub fn run(
    alloc: std.mem.Allocator,
    io: std.Io,
    environ: *const std.process.Environ.Map,
    model_path: []const u8,
    settings: config.Resolved,
    options: Options,
    out: *std.Io.Writer,
    diag: *config.Diagnostic,
) !void {
    const prompt = try engine.readPrompt(alloc, io, options.prompt, options.prompt_file);
    defer alloc.free(prompt);
    const root = try paths.root(alloc, environ.get("NUCLIS_HOME"), environ.get("HOME"));
    defer if (root) |dir| alloc.free(dir);
    const capacity = settings.ctx_size;
    const limit = settings.max_tokens;
    if (capacity == 0 or capacity > config.max_context or limit == 0 or limit > config.max_output_tokens) return error.InvalidGenerationBudget;
    var sampler = try inference.sampling.Sampler.init(options.seed orelse 0, settings.samplingOptions());
    var eng = try engine.Engine.open(alloc, io, model_path, settings.backend, capacity, settings.kv_precision);
    defer eng.deinit();
    const profile = eng.profile orelse return error.UnsupportedPromptTemplate;
    try sampler.setOptions(profile.samplingOptions(settings.think, settings.sampling));
    interrupt.install();

    const logits = try alloc.alloc(f32, eng.vocab.tokens.len);
    defer alloc.free(logits);
    const candidates = try alloc.alloc(inference.sampling.Candidate, eng.vocab.tokens.len);
    defer alloc.free(candidates);
    const generated = try alloc.alloc(u32, limit);
    defer alloc.free(generated);
    var history = try inference.sampling.History.init(alloc, eng.vocab.tokens.len);
    defer history.deinit();

    // The session is opened only when a path was given; `create` with a null
    // root and no override writes nothing at all. The working directory is the
    // tool workspace, canonicalized once.
    var stamp: [20]u8 = undefined;
    const now = model.rfc3339(&stamp, std.Io.Timestamp.now(io, .real).toSeconds());
    var id: [32]u8 = undefined;
    const cwd = std.Io.Dir.cwd().realPathFileAlloc(io, ".", alloc) catch try alloc.dupeZ(u8, ".");
    defer alloc.free(cwd);
    var log = try session_log.create(alloc, io, .cwd(), .{
        .root_dir = null,
        .override_path = options.session,
        .id = session_log.newId(io, &id),
        .time = now,
        .cwd = cwd,
        .model_path = model_path,
        .effort = @tagName(settings.think),
        .ctx_size = capacity,
    });
    defer log.deinit();

    var printer: Printer = .{ .out = out, .json = options.json };
    var turn: Turn = .{ .io = io, .printer = &printer, .log = &log, .started = std.Io.Clock.awake.now(io) };
    var trace: generate.Trace = .{
        .io = io,
        .directory = null,
        .started = turn.started,
        .progress = if (options.json) .{ .context = &turn, .call = Turn.onProgress } else null,
    };
    var completer: loop.Completer = .{
        .alloc = alloc,
        .eng = &eng,
        .effort = settings.think,
        .sampler = &sampler,
        .history = &history,
        .buffers = .{ .logits = logits, .candidates = candidates, .generated = generated, .effort = settings.think },
        .observer = trace.observer(),
    };
    defer completer.deinit();
    const workspace: tools.Workspace = .{ .io = io, .dir = .cwd(), .root = cwd, .environ = environ };
    var agent = try loop.Agent.init(alloc, io, workspace, completer.model(), turn.events(), loop.budget_default);
    defer agent.deinit();
    agent.result_budget = loop.resultBudget(capacity);

    // `--resume <id>`: replay the saved conversation before this turn, so a
    // print run can continue a session. The fresh turn is recorded under
    // `--session`; the replay itself is not, unless that file was named.
    if (options.resume_id) |session_id| {
        // A path needs no root; an id is resolved through the user root.
        const resume_root = root orelse "";
        const path = (try resume_mod.find(alloc, io, resume_root, cwd, session_id)) orelse return error.SessionNotFound;
        defer alloc.free(path);
        var diagnostic: session_log.Diagnostic = .{};
        const loaded = try session_log.load(alloc, io, .cwd(), path, &diagnostic);
        defer loaded.deinit();
        const built = try resume_mod.messages(alloc, loaded);
        defer alloc.free(built);
        try agent.restore(built);
    }

    const stop = agent.turn(prompt) catch |err| {
        if (err == error.ContextFull) {
            const overflow = completer.overflow orelse loop.Overflow{ .needed = 0, .capacity = capacity };
            diag.set("context window full: the step needed {d} tokens (prompt plus output budget) of {d}; raise --ctx-size or shorten the conversation", .{ overflow.needed, overflow.capacity });
        }
        return err;
    };
    if (!options.json and printer.wrote) try out.writeByte('\n');
    try out.flush();
    // A turn can end with no answer at all — the budget spent on reasoning,
    // a cancellation, a call that never resolved — and the text form would
    // then print nothing. Say why on stderr, where it cannot disturb what a
    // script is reading.
    if (!options.json and !printer.wrote) std.log.info("no answer text: stopped on {s}", .{@tagName(stop)});
}

/// What the command line gave print mode. The model, backend, context,
/// budget, effort, and sampling arrive resolved in `config.Resolved`, as for
/// every other command.
pub const Options = struct {
    prompt: ?[]const u8 = null,
    prompt_file: ?[]const u8 = null,
    json: bool = false,
    /// Where to record this turn; null writes no session file.
    session: ?[]const u8 = null,
    /// A saved session id to replay before this turn (`--resume`).
    resume_id: ?[]const u8 = null,
    seed: ?u64 = null,
};

/// The sink the loop drives: terminal events become what print mode writes,
/// and records become session entries.
const Turn = struct {
    io: std.Io,
    printer: *Printer,
    log: *session_log.Session,
    started: std.Io.Timestamp,

    fn events(self: *Turn) loop.Events {
        return .{ .context = self, .send = send, .record = record };
    }

    fn send(context: *anyopaque, e: Event) anyerror!void {
        const self: *Turn = @ptrCast(@alignCast(context));
        try self.printer.send(e);
    }

    fn record(context: *anyopaque, entry: loop.Record) anyerror!void {
        const self: *Turn = @ptrCast(@alignCast(context));
        var stamp: [20]u8 = undefined;
        const now = model.rfc3339(&stamp, std.Io.Timestamp.now(self.io, .real).toSeconds());
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
        try self.log.append(mapped, now);
    }

    fn onProgress(context: *anyopaque, value: inference.observer.Progress) anyerror!void {
        const self: *Turn = @ptrCast(@alignCast(context));
        try self.printer.send(.{ .status = .{
            .phase = switch (value.phase) {
                .prefill => .prefill,
                .decode => .decode,
            },
            .position = value.position,
            .target = value.target,
            .elapsed_ns = @intCast(self.started.durationTo(std.Io.Clock.awake.now(self.io)).nanoseconds),
        } });
    }
};

// ----- tests -----

const testing = std.testing;

test "the text form is the answer, the JSON form is every event" {
    // The printer is the whole presentation layer of print mode, and it needs
    // no model: feeding it the events of a turn is the test.
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    var text_form: Printer = .{ .out = &buffer.writer, .json = false };
    const events = [_]Event{
        .{ .user = "why?" },
        .{ .thinking_delta = "weighing it" },
        .{ .thinking_end = 1.25 },
        .{ .answer_delta = "Because\x07 42" },
        .{ .status = .{ .phase = .decode, .position = 3, .target = 64 } },
        .{ .turn_end = .{ .stop = .eos, .stats = .{ .generated = 3 } } },
    };
    for (events) |e| try text_form.send(e);
    // The answer alone, with the model's control bytes stripped.
    try testing.expectEqualStrings("Because 42", buffer.written());
    try testing.expect(text_form.wrote);

    buffer.clearRetainingCapacity();
    var json_form: Printer = .{ .out = &buffer.writer, .json = true };
    for (events) |e| try json_form.send(e);
    var lines = std.mem.splitScalar(u8, std.mem.trimEnd(u8, buffer.written(), "\n"), '\n');
    var count: usize = 0;
    while (lines.next()) |line| : (count += 1) {
        const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, line, .{});
        defer parsed.deinit();
        try testing.expect(parsed.value.object.get("type") != null);
    }
    try testing.expectEqual(events.len, count);
    try testing.expect(std.mem.indexOf(u8, buffer.written(), "\"type\":\"thinking_delta\"") != null);
    try testing.expect(std.mem.indexOf(u8, buffer.written(), "\"stop\":\"eos\"") != null);
}
