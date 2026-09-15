//! The typed events a turn is made of — the seam between what produces a
//! conversation and what shows it (agent-spec § Transcript and events).
//!
//! Today the producer is the agent's generation loop; in phase 2 it is the
//! engine's event stream and the tool loop, and the consumers are the
//! transcript, the session file, and print mode. All three read the same
//! union, which is why it lives here rather than in `src/agent/`: the
//! terminal surface owns the vocabulary, and it knows nothing about tokens,
//! models, or files.
//!
//! **Ownership deviates from the specification, deliberately.** The spec said
//! strings are owned by the event and freed by the consumer. An `answer_delta`
//! is produced once per generated token, so owning its bytes would mean an
//! allocation and a free per token for text the transcript immediately copies
//! into its own buffer. Here an event **borrows** its strings for the duration
//! of the call: the producer keeps them alive until `apply` returns, and a
//! consumer that keeps text copies it. The one consumer that stores events —
//! the session writer — serializes them immediately, so nothing outlives the
//! call. This is recorded in agent-spec as the step-7 deviation.
const std = @import("std");
const diff = @import("diff.zig");

/// Identifies a tool call so its result can be matched to it. Phase 2 assigns
/// them; phase 1 only carries them through.
pub const Id = u32;

/// Which half of a turn a status beat describes. The engine has the same two
/// names (`inference.observer.Phase`); the agent maps one onto the other, so
/// this file needs no import.
pub const Phase = enum { prefill, decode };

/// Why a turn ended. Mirrors `inference.engine.StopReason`, mapped by the
/// agent for the same reason.
pub const StopReason = enum { eos, token_budget, context_limit, cancelled, failure };

/// Tokens per second, or null when nothing was measured. A rate is never
/// invented: an unavailable measurement stays unavailable all the way to the
/// status bar (the rule `bench` set).
pub const Rates = struct { prefill: ?f64 = null, decode: ?f64 = null };

/// What a finished turn measured, with `bench`'s definitions: prefill covers
/// the turn's new prompt tokens, decode covers `generated - 1` steps.
pub const TurnStats = struct {
    prompt_tokens: usize = 0,
    generated: usize = 0,
    prefill_seconds: f64 = 0,
    decode_seconds: f64 = 0,
    /// The conversation had to be replayed into a fresh session.
    replayed: bool = false,
};

pub const Event = union(enum) {
    /// A committed prompt. Enter emits this before anything can fail.
    user: []const u8,
    /// Text of the reasoning channel, as it arrives.
    thinking_delta: []const u8,
    /// The measured end of the reasoning channel, in seconds from the turn's
    /// start. Not in the specification's union: the transcript has no clock
    /// (it takes no `Io`), so the producer states the measurement instead of
    /// the consumer taking it.
    thinking_end: f64,
    /// Answer text, as it arrives.
    answer_delta: []const u8,
    /// `summary` is the call row (`Reading TODO.md`); `detail` is a row of
    /// its own known at call time (`$ make test`), null for most tools.
    tool_call: struct { id: Id, name: []const u8, summary: []const u8, detail: ?[]const u8 = null },
    /// `text` is what the model reads and the session stores; `summary` is
    /// the one detail row the transcript shows instead, empty when there is
    /// nothing to say.
    tool_result: struct { id: Id, text: []const u8, truncated: bool, is_error: bool, summary: []const u8 = "" },
    /// A mutation's change, as structured rows. The renderer needs no parser:
    /// it lays the rows out (side by side when there is width, unified when
    /// there is not), and the unified text the model and session record comes
    /// from the same computation (`tui.diff`).
    diff: struct { path: []const u8, rows: []const diff.Row },
    status: struct { phase: Phase, position: usize = 0, target: usize = 0, rates: Rates = .{}, elapsed_ns: u64 = 0 },
    turn_end: struct { stop: StopReason, stats: TurnStats = .{} },
    /// Dim system lines: a replay, dropped turns, a failure under a prompt.
    notice: []const u8,
    /// What a command *answered*, as opposed to what it remarked. `/help` is
    /// the first: a page a reader reads, not an aside beside a turn, so it is
    /// rendered at the terminal's own foreground rather than dimmed
    /// (2026-09-12, after the dim help block proved unreadable on a
    /// translucent terminal).
    info: []const u8,

    /// One event as one JSON object, for `nuclis agent --print --json`: the
    /// scripting form of a turn. The tag is the `type` and the payload's fields
    /// sit beside it, so a reader needs no schema beyond the union above. Text
    /// events carry their text under `text`, which is the only name the union
    /// does not already give them.
    ///
    /// This is presentation, not storage: the session file is a different shape
    /// (one entry per turn, not per token) and lives in `src/agent/session.zig`.
    /// Writing to a `std.Io.Writer` is not file I/O, so it belongs here.
    pub fn writeJson(self: Event, out: *std.Io.Writer) !void {
        var s: std.json.Stringify = .{ .writer = out };
        try s.beginObject();
        try s.objectField("type");
        try s.write(@tagName(self));
        switch (self) {
            .user, .thinking_delta, .answer_delta, .notice, .info => |text| {
                try s.objectField("text");
                try s.write(text);
            },
            .thinking_end => |seconds| {
                try s.objectField("seconds");
                try s.write(seconds);
            },
            .diff => |payload| {
                try s.objectField("path");
                try s.write(payload.path);
                try s.objectField("rows");
                try s.beginArray();
                for (payload.rows) |row| {
                    try s.beginObject();
                    try s.objectField("kind");
                    try s.write(@tagName(row.kind));
                    if (row.old_line) |n| {
                        try s.objectField("old");
                        try s.write(n);
                    }
                    if (row.new_line) |n| {
                        try s.objectField("new");
                        try s.write(n);
                    }
                    try s.objectField("text");
                    try s.write(row.text);
                    if (row.change) |span| {
                        try s.objectField("change");
                        try s.beginObject();
                        try s.objectField("start");
                        try s.write(span.start);
                        try s.objectField("len");
                        try s.write(span.len);
                        try s.endObject();
                    }
                    try s.endObject();
                }
                try s.endArray();
            },
            inline .tool_call, .tool_result, .status, .turn_end => |payload| {
                inline for (@typeInfo(@TypeOf(payload)).@"struct".fields) |field| {
                    try s.objectField(field.name);
                    const value = @field(payload, field.name);
                    if (@typeInfo(@TypeOf(value)) == .@"enum") try s.write(@tagName(value)) else try s.write(value);
                }
            },
        }
        try s.endObject();
        try out.writeByte('\n');
    }
};

test "every event kind has a JSON line, with its own fields beside the type" {
    var buffer: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buffer.deinit();
    try (Event{ .answer_delta = "hi\n" }).writeJson(&buffer.writer);
    try std.testing.expectEqualStrings("{\"type\":\"answer_delta\",\"text\":\"hi\\n\"}\n", buffer.written());
    buffer.clearRetainingCapacity();
    try (Event{ .thinking_end = 3.5 }).writeJson(&buffer.writer);
    try std.testing.expectEqualStrings("{\"type\":\"thinking_end\",\"seconds\":3.5}\n", buffer.written());
    buffer.clearRetainingCapacity();
    try (Event{ .status = .{ .phase = .prefill, .position = 768, .target = 1226, .rates = .{ .prefill = 84.25 }, .elapsed_ns = 9 } }).writeJson(&buffer.writer);
    try std.testing.expectEqualStrings(
        "{\"type\":\"status\",\"phase\":\"prefill\",\"position\":768,\"target\":1226,\"rates\":{\"prefill\":84.25,\"decode\":null},\"elapsed_ns\":9}\n",
        buffer.written(),
    );
    buffer.clearRetainingCapacity();
    try (Event{ .turn_end = .{ .stop = .token_budget, .stats = .{ .generated = 400 } } }).writeJson(&buffer.writer);
    try std.testing.expect(std.mem.indexOf(u8, buffer.written(), "\"stop\":\"token_budget\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, buffer.written(), "\"generated\":400") != null);
    buffer.clearRetainingCapacity();
    try (Event{ .tool_call = .{ .id = 3, .name = "grep", .summary = "Searching needle" } }).writeJson(&buffer.writer);
    try std.testing.expectEqualStrings("{\"type\":\"tool_call\",\"id\":3,\"name\":\"grep\",\"summary\":\"Searching needle\",\"detail\":null}\n", buffer.written());
    buffer.clearRetainingCapacity();
    try (Event{ .tool_result = .{ .id = 3, .text = "a:1: needle", .truncated = false, .is_error = false, .summary = "1 match in 1 file" } }).writeJson(&buffer.writer);
    try std.testing.expectEqualStrings("{\"type\":\"tool_result\",\"id\":3,\"text\":\"a:1: needle\",\"truncated\":false,\"is_error\":false,\"summary\":\"1 match in 1 file\"}\n", buffer.written());
    buffer.clearRetainingCapacity();
    const rows = [_]diff.Row{
        .{ .old_line = 1, .new_line = 1, .kind = .context, .text = "keep" },
        .{ .old_line = 2, .kind = .remove, .text = "old", .change = .{ .start = 0, .len = 3 } },
    };
    try (Event{ .diff = .{ .path = "a.zig", .rows = &rows } }).writeJson(&buffer.writer);
    try std.testing.expectEqualStrings(
        "{\"type\":\"diff\",\"path\":\"a.zig\",\"rows\":[{\"kind\":\"context\",\"old\":1,\"new\":1,\"text\":\"keep\"},{\"kind\":\"remove\",\"old\":2,\"text\":\"old\",\"change\":{\"start\":0,\"len\":3}}]}\n",
        buffer.written(),
    );
    // Every kind writes a line and nothing traps on an unusual payload.
    const kinds = [_]Event{
        .{ .user = "u" },                                            .{ .thinking_delta = "t" },
        .{ .thinking_end = 0 },                                      .{ .answer_delta = "a" },
        .{ .tool_call = .{ .id = 1, .name = "n", .summary = "s" } }, .{ .tool_result = .{ .id = 1, .text = "r", .truncated = false, .is_error = false } },
        .{ .diff = .{ .path = "p", .rows = &.{} } },                 .{ .status = .{ .phase = .decode } },
        .{ .turn_end = .{ .stop = .eos } },                          .{ .notice = "n" },
        .{ .info = "i" },
    };
    for (kinds) |kind| {
        buffer.clearRetainingCapacity();
        try kind.writeJson(&buffer.writer);
        try std.testing.expect(std.mem.startsWith(u8, buffer.written(), "{\"type\":\""));
        try std.testing.expect(std.mem.endsWith(u8, buffer.written(), "}\n"));
    }
}

test "an event is a value: no allocation, and the payloads the turn needs" {
    // The union is checked here rather than exercised: its behaviour lives in
    // the transcript (rendering) and the session writer (serialization).
    const events = [_]Event{
        .{ .user = "hello" },
        .{ .thinking_delta = "let me" },
        .{ .thinking_end = 3.4 },
        .{ .answer_delta = "hi" },
        .{ .tool_call = .{ .id = 1, .name = "read_file", .summary = "src/main.zig:1-40" } },
        .{ .tool_result = .{ .id = 1, .text = "…", .truncated = true, .is_error = false } },
        .{ .diff = .{ .path = "src/main.zig", .rows = &.{} } },
        .{ .status = .{ .phase = .prefill, .position = 768, .target = 1226 } },
        .{ .turn_end = .{ .stop = .eos, .stats = .{ .generated = 12 } } },
        .{ .notice = "older turns dropped from context to fit" },
        .{ .info = "keys and commands" },
    };
    try std.testing.expectEqual(@typeInfo(Event).@"union".fields.len, events.len);
    try std.testing.expectEqualStrings("hello", events[0].user);
    try std.testing.expectEqual(@as(f64, 3.4), events[2].thinking_end);
    try std.testing.expect(events[5].tool_result.truncated);
    try std.testing.expectEqual(StopReason.eos, events[8].turn_end.stop);
    // Rates stay optional all the way through: nothing fabricates a number.
    const beat: Event = .{ .status = .{ .phase = .decode, .rates = .{ .decode = 10.5 } } };
    try std.testing.expect(beat.status.rates.prefill == null);
    try std.testing.expectEqual(@as(f64, 10.5), beat.status.rates.decode.?);
}
