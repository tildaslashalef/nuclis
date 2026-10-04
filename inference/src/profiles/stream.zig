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
//! and emits one `tool_call` event when the closing token arrives, with
//! `tool_progress` events while the body grows. A call still open when the
//! completion stops is parsed as if closed at EOS; otherwise (or when it does
//! not parse) it is dropped with a `tool_cut` event: a partial call is
//! neither an action nor answer text. A body that closes but does not parse
//! is released as text, exactly as the model wrote it.
//!
//! The second grammar (`Markers.channel`, Muse Glimmer) is a sequence of
//! messages `HEADER<|message|>BODY` ended by `<|eom|>` or the next
//! `<|start|>`: the header is ordinary text (`assistant to=self`) that
//! routes the body to thinking, the answer, or a tool parser, and a tool
//! body is complete at the end of the turn (`<|eot|>` is a stop token), not
//! at a closing bracket, so `end` completes it on EOS and cuts it on any
//! other stop.
const std = @import("std");
const alloc_check = @import("../alloc_check.zig");
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

/// The message-header grammar: `start` opens a header, `message` ends it
/// and starts the body, `eom` ends a message. `parse` reads a tool body.
pub const Channel = struct {
    start: u32,
    message: u32,
    eom: u32,
    parse: ?ParseTool = null,
};
/// A header longer than this is not one: its bytes are released as answer
/// text (a model writing prose where a header belongs hides nothing).
pub const header_limit = 256;
/// A call body reports its size each time it grows past a multiple of this.
pub const progress_step = 256;

pub const Markers = struct {
    open: u32,
    close: u32,
    /// Gemma's channel token is followed by ordinary `thought\n` text.
    open_suffix: []const u8 = "",
    /// Present only when the template defines a native call grammar.
    tool: ?Tool = null,
    /// Present for the message-header grammar, which replaces the bracket
    /// grammar above (`open`/`close`/`tool` are then unused).
    channel: ?Channel = null,
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
    /// The decoded text of the bracket tokens, copied (a piece is the
    /// caller's per-token buffer), kept so a malformed or truncated call can
    /// be released exactly as the model wrote it.
    tool_open_text: std.ArrayList(u8) = .empty,
    tool_close_text: std.ArrayList(u8) = .empty,
    /// Channel grammar: true while a header is being collected (from the
    /// start of the completion, since the prompt ends inside one).
    in_header: bool = false,
    header_buf: std.ArrayList(u8) = .empty,

    pub fn init(alloc: std.mem.Allocator, markers: Markers, thinking: bool) Decoder {
        const channel = markers.channel != null;
        // Under the channel grammar the first header decides the channel.
        return .{ .alloc = alloc, .markers = markers, .thinking = thinking and !channel, .in_header = channel, .scratch = .init(alloc) };
    }

    pub fn deinit(self: *Decoder) void {
        self.scratch.deinit();
        self.tool_buf.deinit(self.alloc);
        self.tool_open_text.deinit(self.alloc);
        self.tool_close_text.deinit(self.alloc);
        self.header_buf.deinit(self.alloc);
        self.* = undefined;
    }

    /// The completion boundary is shared by the real loop and fake token
    /// sources. Flush text before the single terminal event; finish itself is
    /// also used at channel boundaries and therefore never emits a stop.
    /// Under the channel grammar a tool body open at EOS is a complete call.
    pub fn end(self: *Decoder, outcome: @import("../engine.zig").Outcome, sink: anytype) !void {
        if (self.in_tool) try self.endTool(outcome.stop == .eos, sink);
        try self.finish(sink);
        try sink.send(.{ .stop = outcome });
    }

    /// The token that ends the reasoning once `reasoning` tokens have spent
    /// `budget`, or null: never inside a call or a header being written, and
    /// again after every reopen, so a model that resumes thinking past its
    /// budget writes an empty block instead of unbounded thought.
    pub fn cutToken(self: *const Decoder, reasoning: usize, budget: usize) ?u32 {
        if (!self.thinking or self.in_tool or self.in_header or self.header or reasoning < budget) return null;
        return if (self.markers.channel) |channel| channel.eom else self.markers.close;
    }

    /// A call still open when the completion stopped. At EOS a channel body
    /// is closed by grammar (a call, or text when it does not parse), and a
    /// bracket call missing only its closing token is still one call when it
    /// parses; anything else is dropped and reported.
    fn endTool(self: *Decoder, eos: bool, sink: anytype) !void {
        self.in_tool = false;
        if (eos and self.markers.channel != null) return self.finishChannelTool(sink);
        defer self.tool_buf.clearRetainingCapacity();
        if (eos) {
            if (try self.markers.tool.?.parse(self.alloc, self.tool_buf.items)) |call| {
                defer {
                    self.alloc.free(call.name);
                    self.alloc.free(call.arguments);
                }
                try sink.send(.{ .tool_call = call });
                return;
            }
        }
        self.thinking = false;
        try sink.send(.{ .tool_cut = self.tool_buf.items.len });
    }

    /// Appends a body piece and reports the size when it crosses a step.
    fn collectTool(self: *Decoder, piece: []const u8, sink: anytype) !void {
        const before = self.tool_buf.items.len;
        try self.tool_buf.appendSlice(self.alloc, piece);
        const after = self.tool_buf.items.len;
        if (before / progress_step != after / progress_step or (before == 0 and after != 0)) try sink.send(.{ .tool_progress = after });
    }

    /// `piece` is the decoded bytes of exactly `token`, not accumulated text.
    /// Sink errors propagate immediately; the caller must abandon this decoder.
    pub fn feed(self: *Decoder, token: u32, piece: []const u8, sink: anytype) !void {
        if (self.markers.channel) |channel| return self.feedChannel(channel, token, piece, sink);
        if (self.markers.tool) |tool| {
            if (token == tool.open) {
                try self.finish(sink);
                self.in_tool = true;
                self.tool_buf.clearRetainingCapacity();
                self.tool_open_text.clearRetainingCapacity();
                try self.tool_open_text.appendSlice(self.alloc, piece);
                self.tool_close_text.clearRetainingCapacity();
                return;
            }
            if (self.in_tool and token == tool.close) {
                self.in_tool = false;
                self.tool_close_text.clearRetainingCapacity();
                try self.tool_close_text.appendSlice(self.alloc, piece);
                try self.finishTool(sink);
                return;
            }
        }
        // A call's body is opaque here: it is ordinary text until the closing
        // control token, and the profile parses it as one unit.
        if (self.in_tool) return self.collectTool(piece, sink);
        if (self.thinking and token == self.markers.close) {
            try self.finish(sink);
            self.thinking = false;
            self.trim_answer = true;
            self.initial = false;
            self.header = false;
            self.matched = 0;
            return;
        }
        // A close with no reasoning open (the model ending a thought the
        // engine already closed) is a control token, never answer text.
        if (!self.thinking and token == self.markers.close) {
            self.initial = false;
            return;
        }
        if (self.initial and self.thinking and token == self.markers.open) {
            self.initial = false;
            self.header = self.markers.open_suffix.len != 0;
            return;
        }
        // The channel opened (again) while answering: thinking from here,
        // whatever the effort — a model with thinking off still opens an
        // empty channel after a tool result. The transcript shows a block;
        // the alternative, the marker as answer text, would be re-encoded
        // as a control token next turn.
        if (!self.thinking and token == self.markers.open) {
            try self.finish(sink);
            self.thinking = true;
            self.initial = false;
            self.header = self.markers.open_suffix.len != 0;
            self.matched = 0;
            self.trim_answer = false;
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

    fn feedChannel(self: *Decoder, channel: Channel, token: u32, piece: []const u8, sink: anytype) !void {
        if (token == channel.start) {
            try self.closeMessage(sink);
            self.in_header = true;
            self.header_buf.clearRetainingCapacity();
            return;
        }
        if (self.in_header) {
            if (token == channel.message) {
                self.in_header = false;
                self.route(channel);
                return;
            }
            try self.header_buf.appendSlice(self.alloc, piece);
            if (self.header_buf.items.len > header_limit) {
                // Not a header after all: nothing stays hidden.
                self.in_header = false;
                self.thinking = false;
                try self.text(self.header_buf.items, sink);
                self.header_buf.clearRetainingCapacity();
            }
            return;
        }
        if (token == channel.eom) {
            try self.closeMessage(sink);
            self.in_header = true;
            self.header_buf.clearRetainingCapacity();
            return;
        }
        if (self.in_tool) return self.collectTool(piece, sink);
        try self.text(piece, sink);
    }

    /// The header decides the body's channel: `assistant to=self` is
    /// thinking, `assistant` and `assistant to=user` the answer, any other
    /// recipient a tool body when the profile parses one, else the answer.
    fn route(self: *Decoder, channel: Channel) void {
        var header = std.mem.trim(u8, self.header_buf.items, " \t\r\n");
        if (std.mem.startsWith(u8, header, "assistant")) header = std.mem.trimStart(u8, header["assistant".len..], " \t\r\n");
        self.thinking = false;
        self.in_tool = false;
        if (std.mem.eql(u8, header, "to=self")) {
            self.thinking = true;
        } else if (header.len != 0 and !std.mem.eql(u8, header, "to=user") and std.mem.startsWith(u8, header, "to=") and channel.parse != null) {
            self.in_tool = true;
            self.tool_buf.clearRetainingCapacity();
        }
        self.header_buf.clearRetainingCapacity();
    }

    /// A message boundary under the channel grammar: a tool body is one
    /// complete call, other text is flushed with its UTF-8 state.
    fn closeMessage(self: *Decoder, sink: anytype) !void {
        if (self.in_tool) {
            self.in_tool = false;
            try self.finishChannelTool(sink);
            return;
        }
        self.scratch.clearRetainingCapacity();
        try self.utf8.finish(&self.scratch.writer);
        try self.flush(sink);
    }

    /// Parse and emit the collected tool body, or release it as answer text.
    fn finishChannelTool(self: *Decoder, sink: anytype) !void {
        const parse = self.markers.channel.?.parse.?;
        const parsed = try parse(self.alloc, self.tool_buf.items);
        if (parsed) |call| {
            defer {
                self.alloc.free(call.name);
                self.alloc.free(call.arguments);
            }
            try sink.send(.{ .tool_call = call });
        } else {
            self.thinking = false;
            try self.text(self.tool_buf.items, sink);
        }
        self.tool_buf.clearRetainingCapacity();
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
            try self.text(self.tool_open_text.items, sink);
            try self.text(self.tool_buf.items, sink);
            try self.text(self.tool_close_text.items, sink);
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

    /// Release incomplete header text and an unfinished UTF-8 scalar at a
    /// stop or a channel boundary. An open call is `end`'s to settle; a
    /// boundary inside a call is not one.
    pub fn finish(self: *Decoder, sink: anytype) !void {
        std.debug.assert(!self.in_tool);
        if (self.in_header) {
            self.in_header = false;
            self.thinking = false;
            try self.text(self.header_buf.items, sink);
            self.header_buf.clearRetainingCapacity();
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

test "thinking off keeps leading answer newlines and drops a stray close" {
    var r = Recorder.init(std.testing.allocator);
    defer r.deinit();
    var d = Decoder.init(std.testing.allocator, .{ .open = 1, .close = 2 }, false);
    defer d.deinit();
    try d.feed(3, "\nanswer", &r);
    // A control token, not text: shown, it would be re-encoded as one.
    try d.feed(2, "</think>", &r);
    try d.finish(&r);
    try std.testing.expectEqualStrings("", r.thinking.written());
    try std.testing.expectEqualStrings("\nanswer", r.answer.written());
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
    /// The last `tool_progress` size and the `tool_cut` size, if any.
    progress: usize = 0,
    progress_events: usize = 0,
    cut: ?usize = null,
    stops: usize = 0,

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
            .tool_progress => |bytes| {
                self.progress = bytes;
                self.progress_events += 1;
            },
            .tool_cut => |bytes| self.cut = bytes,
            .stop => self.stops += 1,
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

test "a channel opened while answering is thinking again, whatever the effort" {
    var r = Recorder.init(std.testing.allocator);
    defer r.deinit();
    var d = Decoder.init(std.testing.allocator, .{ .open = 1, .close = 2, .open_suffix = "thought\n" }, true);
    defer d.deinit();
    try d.feed(1, "<|channel>", &r);
    try d.feed(3, "thought\nplan", &r);
    try d.feed(2, "<channel|>", &r);
    try d.feed(3, "\nreply", &r);
    try d.feed(1, "<|channel>", &r);
    try d.feed(3, "thought\nmore", &r);
    try d.feed(2, "<channel|>", &r);
    try d.feed(3, "\nend", &r);
    try d.finish(&r);
    try std.testing.expectEqualStrings("planmore", r.thinking.written());
    try std.testing.expectEqualStrings("replyend", r.answer.written());
    // An empty reopened channel adds nothing anywhere.
    var e = Recorder.init(std.testing.allocator);
    defer e.deinit();
    var empty = Decoder.init(std.testing.allocator, .{ .open = 1, .close = 2, .open_suffix = "thought\n" }, true);
    defer empty.deinit();
    try empty.feed(2, "<channel|>", &e);
    try empty.feed(3, "a", &e);
    try empty.feed(1, "<|channel>", &e);
    try empty.feed(3, "thought\n", &e);
    try empty.feed(2, "<channel|>", &e);
    try empty.feed(3, "b", &e);
    try empty.finish(&e);
    try std.testing.expectEqualStrings("", e.thinking.written());
    try std.testing.expectEqualStrings("ab", e.answer.written());
    // With reasoning off for the turn an opening token still opens a
    // channel (a close without one stays literal: the test above).
    var o = Recorder.init(std.testing.allocator);
    defer o.deinit();
    var off = Decoder.init(std.testing.allocator, .{ .open = 1, .close = 2, .open_suffix = "thought\n" }, false);
    defer off.deinit();
    try off.feed(3, "a", &o);
    try off.feed(1, "<|channel>", &o);
    try off.feed(3, "thought\n", &o);
    try off.feed(2, "<channel|>", &o);
    try off.feed(3, "b", &o);
    try off.finish(&o);
    try std.testing.expectEqualStrings("", o.thinking.written());
    try std.testing.expectEqualStrings("ab", o.answer.written());
}

test "the bracket pieces are copied, so a released call shows the marker the model wrote" {
    var r = ToolSink.init(std.testing.allocator);
    defer r.deinit();
    var d = toolDecoder();
    defer d.deinit();
    // The caller reuses one buffer per token: after the open bracket it
    // holds the body's bytes.
    var buffer: [11]u8 = "<tool_call>".*;
    try d.feed(3, &buffer, &r);
    @memset(&buffer, 0xff);
    try d.feed(11, "BO", &r);
    try d.feed(4, "</tool_call>", &r);
    try d.finish(&r);
    try std.testing.expectEqualStrings("<tool_call>BO</tool_call>", r.answer.written());
}

test "a call open at a stop is cut, not text; at EOS a whole body is a call; a rejected body is text" {
    const engine = @import("../engine.zig");
    var r = ToolSink.init(std.testing.allocator);
    defer r.deinit();
    // Cancelled (or out of budget) before the closing control: dropped and
    // reported, never shown as the model's answer.
    var truncated = toolDecoder();
    defer truncated.deinit();
    try truncated.feed(3, "<tool_call>", &r);
    try truncated.feed(11, "BO", &r);
    try truncated.end(.{ .stop = .cancelled, .timing = .{} }, &r);
    try std.testing.expectEqual(@as(usize, 0), r.calls);
    try std.testing.expectEqual(@as(?usize, 2), r.cut);
    try std.testing.expectEqualStrings("", r.answer.written());

    // EOS with a body that parses: the model only left out the close.
    var unclosed = toolDecoder();
    defer unclosed.deinit();
    try unclosed.feed(3, "<tool_call>", &r);
    try unclosed.feed(11, "BODY", &r);
    try unclosed.end(@as(engine.Outcome, .{ .stop = .eos, .timing = .{} }), &r);
    try std.testing.expectEqual(@as(usize, 1), r.calls);

    // Malformed: the body arrives but the parser refuses it.
    r.answer.clearRetainingCapacity();
    r.calls = 0;
    var malformed = toolDecoder();
    defer malformed.deinit();
    try malformed.feed(3, "<tool_call>", &r);
    try malformed.feed(11, "\nNOPE\n", &r);
    try malformed.feed(4, "</tool_call>", &r);
    try malformed.finish(&r);
    try std.testing.expectEqual(@as(usize, 0), r.calls);
    try std.testing.expectEqualStrings("<tool_call>\nNOPE\n</tool_call>", r.answer.written());
}

test "a growing call body reports its size; a stray close while answering is dropped" {
    var r = ToolSink.init(std.testing.allocator);
    defer r.deinit();
    var d = toolDecoder();
    defer d.deinit();
    try d.feed(10, "plan", &r);
    // The engine closed nothing here: the model's own close is no text.
    try d.feed(2, "</think>", &r);
    try d.feed(10, " done", &r);
    try d.feed(3, "<tool_call>", &r);
    const piece: [100]u8 = @splat('x');
    for (0..6) |_| try d.feed(11, &piece, &r);
    // 100 bytes (the first piece), then 300 and 600 cross 256 and 512.
    try std.testing.expectEqual(@as(usize, 3), r.progress_events);
    try std.testing.expectEqual(@as(usize, 600), r.progress);
    try d.end(.{ .stop = .token_budget, .timing = .{} }, &r);
    try std.testing.expectEqualStrings("plan done", r.answer.written());
    try std.testing.expectEqual(@as(?usize, 600), r.cut);
}

test "past the budget every reopened reasoning block is cut again, after its header" {
    var r = Recorder.init(std.testing.allocator);
    defer r.deinit();
    var d = Decoder.init(std.testing.allocator, .{ .open = 1, .close = 2, .open_suffix = "thought\n" }, true);
    defer d.deinit();
    try std.testing.expectEqual(@as(?u32, null), d.cutToken(9, 10));
    try d.feed(1, "<|channel>", &r);
    try std.testing.expectEqual(@as(?u32, null), d.cutToken(10, 10)); // the header is still due
    try d.feed(3, "thought\nplan", &r);
    try std.testing.expectEqual(@as(?u32, 2), d.cutToken(10, 10));
    try d.feed(2, "<channel|>", &r);
    try std.testing.expectEqual(@as(?u32, null), d.cutToken(11, 10)); // answering
    // The model opens its reasoning again: closed again once its header is in.
    try d.feed(1, "<|channel>", &r);
    try std.testing.expectEqual(@as(?u32, null), d.cutToken(11, 10));
    try d.feed(3, "thought\n", &r);
    try std.testing.expectEqual(@as(?u32, 2), d.cutToken(11, 10));
    try d.feed(2, "<channel|>", &r);
    try d.feed(3, "answer", &r);
    try std.testing.expectEqualStrings("plan", r.thinking.written());
    try std.testing.expectEqualStrings("answer", r.answer.written());
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
    try alloc_check.checkAll(std.testing.allocator, allocationPaths, .{});
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
