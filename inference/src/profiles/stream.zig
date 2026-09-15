//! Incremental profile decoding, before token boundaries are lost.
//!
//! The two text profiles share a channel state machine, configured by each
//! profile's actual control tokens. A marker spelled with ordinary tokens is
//! ordinary text. Only the channel header's non-special suffix needs a small
//! hold-back; reasoning and answer content stream without marker-sized delays.
//! UTF-8 repair is local to each channel, so a partial scalar cannot cross a
//! control token. Scratch storage is reused per piece, never per completion.
//!
//! A profile whose template defines tool calling supplies `Markers.tool`: the
//! two control tokens that bracket a call and a parser for the body between
//! them. The decoder recognises those tokens by id (the only structural signal
//! a byte-stream consumer cannot see), collects the body as ordinary pieces,
//! and emits one `tool_call` event when the closing token arrives. A call that
//! is still open at EOS, a budget stop, or a cancellation is released as plain
//! answer text and never surfaces for execution.
const std = @import("std");
const Utf8 = @import("../tokenizer/stream.zig").Stream;
const ToolCall = @import("../events.zig").ToolCall;

/// A template's native call grammar: the bracket tokens and the body parser.
/// `parse` returns an owned call (the slices live until the caller frees
/// them), or null for a body that is not one complete, well-formed call.
pub const ParseTool = *const fn (alloc: std.mem.Allocator, body: []const u8) std.mem.Allocator.Error!?ToolCall;
pub const Tool = struct {
    open: u32,
    close: u32,
    parse: ParseTool,
};

pub const Markers = struct {
    open: u32,
    close: u32,
    /// Gemma's channel token is followed by ordinary `thought\n` text.
    open_suffix: []const u8 = "",
    /// Present only when the template defines a native call grammar.
    tool: ?Tool = null,
};

pub const Decoder = struct {
    alloc: std.mem.Allocator,
    markers: Markers,
    thinking: bool,
    initial: bool = true,
    header: bool = false,
    matched: usize = 0,
    trim_answer: bool = false,
    utf8: Utf8 = .{},
    scratch: std.Io.Writer.Allocating,
    /// True between the opening and closing call-control tokens.
    in_tool: bool = false,
    /// The body of the call being collected, ordinary pieces only.
    tool_buf: std.ArrayList(u8) = .empty,
    /// The decoded text of the bracket tokens, kept so a malformed or
    /// truncated call can be released exactly as the model wrote it.
    tool_open_text: []const u8 = "",
    tool_close_text: []const u8 = "",

    pub fn init(alloc: std.mem.Allocator, markers: Markers, thinking: bool) Decoder {
        return .{ .alloc = alloc, .markers = markers, .thinking = thinking, .scratch = .init(alloc) };
    }

    pub fn deinit(self: *Decoder) void {
        self.scratch.deinit();
        self.tool_buf.deinit(self.alloc);
        self.* = undefined;
    }

    /// The completion boundary is shared by the real loop and fake token
    /// sources. Flush text before the single terminal event; finish itself is
    /// also used at channel boundaries and therefore never emits a stop.
    pub fn end(self: *Decoder, outcome: @import("../engine.zig").Outcome, sink: anytype) !void {
        try self.finish(sink);
        try sink.send(.{ .stop = outcome });
    }

    /// `piece` is the decoded bytes of exactly `token`, not accumulated text.
    /// Sink errors propagate immediately; the caller must abandon this decoder.
    pub fn feed(self: *Decoder, token: u32, piece: []const u8, sink: anytype) !void {
        if (self.markers.tool) |tool| {
            if (token == tool.open) {
                try self.finish(sink);
                self.in_tool = true;
                self.tool_buf.clearRetainingCapacity();
                self.tool_open_text = piece;
                self.tool_close_text = "";
                return;
            }
            if (self.in_tool and token == tool.close) {
                self.in_tool = false;
                self.tool_close_text = piece;
                try self.finishTool(sink);
                return;
            }
        }
        // A call's body is opaque here: it is ordinary text until the closing
        // control token, and the profile parses it as one unit.
        if (self.in_tool) {
            try self.tool_buf.appendSlice(self.alloc, piece);
            return;
        }
        if (self.thinking and token == self.markers.close) {
            try self.finish(sink);
            self.thinking = false;
            self.trim_answer = true;
            self.initial = false;
            return;
        }
        if (self.initial and self.thinking and token == self.markers.open) {
            self.initial = false;
            self.header = self.markers.open_suffix.len != 0;
            return;
        }
        self.initial = false;
        var bytes = piece;
        if (self.header) {
            while (bytes.len != 0 and self.matched < self.markers.open_suffix.len) {
                if (bytes[0] != self.markers.open_suffix[self.matched]) {
                    self.header = false;
                    try self.text(self.markers.open_suffix[0..self.matched], sink);
                    self.matched = 0;
                    break;
                }
                self.matched += 1;
                bytes = bytes[1..];
            }
            if (self.matched == self.markers.open_suffix.len) {
                self.header = false;
                self.matched = 0;
            }
            if (self.header) return;
        }
        if (self.trim_answer) {
            bytes = std.mem.trimStart(u8, bytes, "\n");
            if (bytes.len != 0) self.trim_answer = false;
        }
        try self.text(bytes, sink);
    }

    /// One complete call: parse the collected body and emit it, or release the
    /// whole thing as answer text when it is not one well-formed call.
    fn finishTool(self: *Decoder, sink: anytype) !void {
        const tool = self.markers.tool.?;
        const parsed = try tool.parse(self.alloc, self.tool_buf.items);
        if (parsed) |call| {
            defer {
                self.alloc.free(call.name);
                self.alloc.free(call.arguments);
            }
            try sink.send(.{ .tool_call = call });
        } else {
            try self.text(self.tool_open_text, sink);
            try self.text(self.tool_buf.items, sink);
            try self.text(self.tool_close_text, sink);
        }
        self.tool_buf.clearRetainingCapacity();
    }

    fn text(self: *Decoder, bytes: []const u8, sink: anytype) !void {
        self.scratch.clearRetainingCapacity();
        try self.utf8.write(bytes, &self.scratch.writer);
        try self.flush(sink);
    }

    fn flush(self: *Decoder, sink: anytype) !void {
        const bytes = self.scratch.written();
        if (bytes.len == 0) return;
        if (self.thinking) try sink.send(.{ .thinking = bytes }) else try sink.send(.{ .answer = bytes });
    }

    /// Release incomplete header text, an open call, and an unfinished UTF-8
    /// scalar at EOS, budget, context limit, cancellation, or a channel
    /// boundary. A call still open here never reached its closing token, so it
    /// is text, not an action.
    pub fn finish(self: *Decoder, sink: anytype) !void {
        if (self.in_tool) {
            self.in_tool = false;
            try self.text(self.tool_open_text, sink);
            try self.text(self.tool_buf.items, sink);
            self.tool_buf.clearRetainingCapacity();
        }
        if (self.header) {
            self.header = false;
            try self.text(self.markers.open_suffix[0..self.matched], sink);
            self.matched = 0;
        }
        self.scratch.clearRetainingCapacity();
        try self.utf8.finish(&self.scratch.writer);
        try self.flush(sink);
    }
};

// A fake token source drives the exact decoder/sink boundary, without a model,
// GPU, clock, or tokenizer. IDs deliberately differ from production IDs.
const Event = @import("../events.zig").Event;
const Recorder = struct {
    thinking: std.Io.Writer.Allocating,
    answer: std.Io.Writer.Allocating,
    calls: usize = 0,

    fn init(alloc: std.mem.Allocator) Recorder {
        return .{ .thinking = .init(alloc), .answer = .init(alloc) };
    }
    fn deinit(self: *Recorder) void {
        self.thinking.deinit();
        self.answer.deinit();
    }
    pub fn send(self: *Recorder, event: Event) !void {
        self.calls += 1;
        switch (event) {
            .thinking => |text| try self.thinking.writer.writeAll(text),
            .answer => |text| try self.answer.writer.writeAll(text),
            else => return error.UnexpectedEvent,
        }
    }
};

test "only actual control IDs split reasoning; ordinary marker text stays literal" {
    var r = Recorder.init(std.testing.allocator);
    defer r.deinit();
    var d = Decoder.init(std.testing.allocator, .{ .open = 1, .close = 2 }, true);
    defer d.deinit();
    try d.feed(1, "<think>", &r);
    try d.feed(3, "work", &r);
    try std.testing.expectEqualStrings("work", r.thinking.written()); // no hold-back
    try d.feed(3, "</thi", &r);
    try d.feed(3, "nk>", &r);
    try d.feed(2, "</think>", &r);
    try d.feed(3, "\n", &r);
    try d.feed(3, "\nanswer</think>", &r);
    try d.finish(&r);
    try std.testing.expectEqualStrings("work</think>", r.thinking.written());
    try std.testing.expectEqualStrings("answer</think>", r.answer.written());
}

test "Gemma's ordinary channel header suffix can split at every byte" {
    const text = "thought\nplan";
    for (0..text.len + 1) |split| {
        var r = Recorder.init(std.testing.allocator);
        defer r.deinit();
        var d = Decoder.init(std.testing.allocator, .{ .open = 1, .close = 2, .open_suffix = "thought\n" }, true);
        defer d.deinit();
        try d.feed(1, "<|channel>", &r);
        try d.feed(3, text[0..split], &r);
        try d.feed(3, text[split..], &r);
        try d.feed(2, "<channel|>", &r);
        try d.feed(3, "reply", &r);
        try d.finish(&r);
        try std.testing.expectEqualStrings("plan", r.thinking.written());
        try std.testing.expectEqualStrings("reply", r.answer.written());
    }
}

test "UTF-8 split at every byte, cancelled thought, and no cross-channel scalar" {
    const text = "aé中🙂z";
    for (0..text.len + 1) |split| {
        var r = Recorder.init(std.testing.allocator);
        defer r.deinit();
        var d = Decoder.init(std.testing.allocator, .{ .open = 1, .close = 2 }, true);
        defer d.deinit();
        try d.feed(3, text[0..split], &r);
        try d.feed(3, text[split..], &r);
        try d.finish(&r); // budget/cancellation while still thinking
        try std.testing.expectEqualStrings(text, r.thinking.written());
        try std.testing.expectEqualStrings("", r.answer.written());
    }
    var r = Recorder.init(std.testing.allocator);
    defer r.deinit();
    var d = Decoder.init(std.testing.allocator, .{ .open = 1, .close = 2 }, true);
    defer d.deinit();
    try d.feed(3, "\xe2", &r);
    try d.feed(2, "</think>", &r);
    try d.feed(3, "\x82\xac", &r);
    try d.finish(&r);
    try std.testing.expectEqualStrings("�", r.thinking.written());
    try std.testing.expectEqualStrings("��", r.answer.written());
}

test "thinking off preserves marker text and leading answer newlines" {
    var r = Recorder.init(std.testing.allocator);
    defer r.deinit();
    var d = Decoder.init(std.testing.allocator, .{ .open = 1, .close = 2 }, false);
    defer d.deinit();
    try d.feed(3, "\nanswer", &r);
    try d.feed(2, "</think>", &r);
    try d.finish(&r);
    try std.testing.expectEqualStrings("", r.thinking.written());
    try std.testing.expectEqualStrings("\nanswer</think>", r.answer.written());
}

test "empty completion, partial header, and malformed header release no hidden bytes" {
    var r = Recorder.init(std.testing.allocator);
    defer r.deinit();
    var d = Decoder.init(std.testing.allocator, .{ .open = 1, .close = 2, .open_suffix = "thought\n" }, true);
    defer d.deinit();
    try d.finish(&r);
    try std.testing.expectEqual(@as(usize, 0), r.calls);
    try d.feed(1, "<|channel>", &r);
    try d.feed(3, "thou", &r);
    try d.finish(&r);
    try std.testing.expectEqualStrings("thou", r.thinking.written());
    var other = Decoder.init(std.testing.allocator, .{ .open = 1, .close = 2, .open_suffix = "thought\n" }, true);
    defer other.deinit();
    try other.feed(1, "<|channel>", &r);
    try other.feed(3, "th", &r);
    try other.feed(3, "e plan", &r);
    try other.finish(&r);
    try std.testing.expectEqualStrings("thouthe plan", r.thinking.written());
}

/// A tool-syntax test parser and a sink that retains the emitted call, so the
/// decode path can be exercised without a model or a tokenizer.
const ToolFixture = struct {
    fn parse(alloc: std.mem.Allocator, body: []const u8) std.mem.Allocator.Error!?ToolCall {
        if (!std.mem.eql(u8, std.mem.trim(u8, body, "\n"), "BODY")) return null;
        const name = try alloc.dupe(u8, "read_file");
        errdefer alloc.free(name);
        return .{ .id = 0, .name = name, .arguments = try alloc.dupe(u8, "{\"path\":\"a\"}") };
    }
};

const ToolSink = struct {
    alloc: std.mem.Allocator,
    answer: std.Io.Writer.Allocating,
    name: std.Io.Writer.Allocating,
    arguments: std.Io.Writer.Allocating,
    calls: usize = 0,

    fn init(alloc: std.mem.Allocator) ToolSink {
        return .{ .alloc = alloc, .answer = .init(alloc), .name = .init(alloc), .arguments = .init(alloc) };
    }
    fn deinit(self: *ToolSink) void {
        self.answer.deinit();
        self.name.deinit();
        self.arguments.deinit();
    }
    fn send(self: *ToolSink, event: Event) !void {
        switch (event) {
            .answer => |text| try self.answer.writer.writeAll(text),
            .tool_call => |call| {
                self.calls += 1;
                self.name.clearRetainingCapacity();
                self.arguments.clearRetainingCapacity();
                try self.name.writer.writeAll(call.name);
                try self.arguments.writer.writeAll(call.arguments);
            },
            else => return error.UnexpectedEvent,
        }
    }
};

fn toolDecoder() Decoder {
    return Decoder.init(std.testing.allocator, .{
        .open = 1,
        .close = 2,
        .tool = .{ .open = 3, .close = 4, .parse = ToolFixture.parse },
    }, false);
}

test "a bracketed call is collected across pieces and emitted once" {
    var r = ToolSink.init(std.testing.allocator);
    defer r.deinit();
    var d = toolDecoder();
    defer d.deinit();
    try d.feed(10, "before ", &r);
    try d.feed(3, "<tool_call>", &r);
    // The body is ordinary text; a token can split it at any byte.
    for ("BODY") |byte| try d.feed(11, &.{byte}, &r);
    try d.feed(4, "</tool_call>", &r);
    try d.feed(10, " after", &r);
    try d.finish(&r);
    try std.testing.expectEqualStrings("before  after", r.answer.written());
    try std.testing.expectEqual(@as(usize, 1), r.calls);
    try std.testing.expectEqualStrings("read_file", r.name.written());
    try std.testing.expectEqualStrings("{\"path\":\"a\"}", r.arguments.written());
}

test "a call still open at finish, or rejecting its body, is released as text" {
    var r = ToolSink.init(std.testing.allocator);
    defer r.deinit();
    // Truncation: the closing control never arrives.
    var truncated = toolDecoder();
    defer truncated.deinit();
    try truncated.feed(3, "<tool_call>", &r);
    try truncated.feed(11, "BO", &r);
    try truncated.finish(&r);
    try std.testing.expectEqual(@as(usize, 0), r.calls);
    try std.testing.expectEqualStrings("<tool_call>BO", r.answer.written());

    // Malformed: the body arrives but the parser refuses it.
    r.answer.clearRetainingCapacity();
    var malformed = toolDecoder();
    defer malformed.deinit();
    try malformed.feed(3, "<tool_call>", &r);
    try malformed.feed(11, "\nNOPE\n", &r);
    try malformed.feed(4, "</tool_call>", &r);
    try malformed.finish(&r);
    try std.testing.expectEqual(@as(usize, 0), r.calls);
    try std.testing.expectEqualStrings("<tool_call>\nNOPE\n</tool_call>", r.answer.written());
}

fn allocationPaths(alloc: std.mem.Allocator) !void {
    // Every writer in this test is allocating: its WriteFailed means OOM.
    allocationCase(alloc) catch |err| return switch (err) {
        error.WriteFailed => error.OutOfMemory,
        else => err,
    };
}

fn allocationCase(alloc: std.mem.Allocator) !void {
    var r = Recorder.init(alloc);
    defer r.deinit();
    var d = Decoder.init(alloc, .{ .open = 1, .close = 2 }, true);
    defer d.deinit();
    try d.feed(3, "plan", &r);
    try d.feed(2, "</think>", &r);
    try d.feed(3, "answer", &r);
    try d.finish(&r);
}

test "decoder and retaining sink clean up every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationPaths, .{});
}

test "sink failure propagates without further events" {
    const Failed = struct {
        calls: usize = 0,
        pub fn send(self: *@This(), _: Event) !void {
            self.calls += 1;
            return error.SinkFailed;
        }
    };
    var sink: Failed = .{};
    var d = Decoder.init(std.testing.allocator, .{ .open = 1, .close = 2 }, false);
    defer d.deinit();
    try std.testing.expectError(error.SinkFailed, d.feed(3, "answer", &sink));
    try std.testing.expectEqual(@as(usize, 1), sink.calls);
}

test "stubbed completions flush text then emit one stop with unchanged metrics" {
    const engine = @import("../engine.zig");
    const Sink = struct {
        text: std.Io.Writer.Allocating,
        stop: ?engine.Outcome = null,
        stops: usize = 0,
        pub fn send(self: *@This(), e: Event) !void {
            try std.testing.expect(self.stop == null);
            switch (e) {
                .answer, .thinking => |bytes| try self.text.writer.writeAll(bytes),
                .stop => |outcome| {
                    self.stop = outcome;
                    self.stops += 1;
                },
                else => return error.UnexpectedEvent,
            }
        }
    };
    for ([_]engine.StopReason{ .eos, .token_budget, .context_limit, .cancelled }) |reason| {
        for ([_]bool{ false, true }) |empty| {
            var sink: Sink = .{ .text = .init(std.testing.allocator) };
            defer sink.text.deinit();
            var decoder = Decoder.init(std.testing.allocator, .{ .open = 1, .close = 2 }, true);
            defer decoder.deinit();
            // An incomplete scalar tests the final flush, including a cancel
            // before any answer. Empty covers cancellation during prefill.
            if (!empty) try decoder.feed(3, "plan\xe2", &sink);
            const outcome: engine.Outcome = .{ .stop = reason, .timing = .{ .prompt_tokens = 4, .generated_tokens = if (empty) 0 else 1 } };
            try decoder.end(outcome, &sink);
            try std.testing.expectEqualStrings(if (empty) "" else "plan�", sink.text.written());
            try std.testing.expectEqualDeep(outcome, sink.stop.?);
            try std.testing.expectEqual(@as(usize, 1), sink.stops);
        }
    }
}
