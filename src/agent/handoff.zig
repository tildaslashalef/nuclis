//! The hand-off: what replaces a conversation that no longer fits the
//! window once the cheap cuts are spent (`loop.Agent.handoff`), or on
//! `/compact`. It is one user message: a ledger the host builds from what
//! it observed (every request, the answer the user replied to, the files
//! written, the last failing command), then a summary the model writes over
//! a condensed render of the conversation, then the request in progress.
//! The pieces here are pure; the loop owns the model step and the history.
//! The contract: docs/app/agent-surface.md § The loop (Compaction).
const std = @import("std");
const Allocator = std.mem.Allocator;

/// The hand-off message's first line, which tells the model what it reads.
pub const header = "[hand-off: the conversation so far was replaced by this summary to fit the context]";

/// The summary step's system block: short, since it is rendered alone.
pub const system = "You write hand-off notes for a coding session, read by whoever continues it with no other memory of it.";

/// What the summary step is asked, after the condensed conversation.
pub const instruction =
    \\Write the hand-off now, under exactly these headings: Goal, Done, Current state, Decisions, Next step, Open problems.
    \\Be specific: name files, functions, commands, and error messages. At most 400 words. Do not call tools.
;

/// Each request kept in the ledger, cut to this many bytes.
pub const ask_bytes = 2048;
/// Requests the ledger keeps: the first `first_asks`, then the latest.
pub const max_asks = 16;
const first_asks = 4;
/// The answer the user replied to, cut to this many bytes.
pub const agreed_bytes = 4096;
/// Files the ledger lists, most recent last.
pub const max_touched = 64;
/// Lines of the last failing command's output the ledger keeps.
pub const failing_lines = 40;
/// The condensed conversation's bound in tokens; the window may lower it.
pub const condensed_tokens = 12 * 1024;
/// Each answer and each result line in the condensed render, cut to this.
pub const condensed_answer_bytes = 1024;
pub const condensed_result_bytes = 200;
/// The last results kept whole in the condensed render, each cut to this.
pub const condensed_full_results = 2;
pub const condensed_full_bytes = 4096;
/// Write and edit arguments at or under this size are kept when earlier
/// calls are shortened: a small edit is cheap and says what changed.
pub const stub_min_bytes = 512;

/// What a shortened argument says, so a later pass knows it is done.
pub const stub_marker = "[shortened to fit the context: ";

/// A file the session wrote or edited, and the last result's summary line.
pub const Touched = struct { path: []const u8, summary: []const u8 };

/// What the host observed, for `write`. Borrowed.
pub const Ledger = struct {
    asks: []const []const u8 = &.{},
    agreed: []const u8 = "",
    touched: []const Touched = &.{},
    failing: []const u8 = "",
};

/// `text` cut to at most `max` bytes at a code point, or whole.
pub fn cut(text: []const u8, max: usize) []const u8 {
    if (text.len <= max) return text;
    var end = max;
    while (end > 0 and (text[end] & 0xc0) == 0x80) end -= 1;
    return text[0..end];
}

/// The last `n` lines of `text` (borrowed).
pub fn tailLines(text: []const u8, n: usize) []const u8 {
    const trimmed = std.mem.trimEnd(u8, text, "\n");
    var seen: usize = 0;
    var i = trimmed.len;
    while (i > 0) {
        i -= 1;
        if (trimmed[i] == '\n') {
            seen += 1;
            if (seen == n) return trimmed[i + 1 ..];
        }
    }
    return trimmed;
}

/// The arguments an earlier `write_file` or `edit_file` call keeps once
/// shortened: its path and a stub naming what the bulk was, in the same
/// JSON shape, so the profile renders it as an ordinary call. Null when
/// the call is another tool's, already short, already shortened, or not
/// an object with a path. Caller owns the result.
pub fn argumentStub(alloc: Allocator, name: []const u8, arguments: []const u8) !?[]u8 {
    const write_file = std.mem.eql(u8, name, "write_file");
    if (!write_file and !std.mem.eql(u8, name, "edit_file")) return null;
    if (arguments.len <= stub_min_bytes or std.mem.indexOf(u8, arguments, stub_marker) != null) return null;
    const Args = struct { path: []const u8, content: []const u8 = "", new_string: []const u8 = "", edits: ?std.json.Value = null };
    const parsed = std.json.parseFromSlice(Args, alloc, arguments, .{ .ignore_unknown_fields = true }) catch return null;
    defer parsed.deinit();
    const args = parsed.value;
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    var s: std.json.Stringify = .{ .writer = &out.writer };
    var note: [96]u8 = undefined;
    try s.beginObject();
    try s.objectField("path");
    try s.write(args.path);
    if (write_file) {
        try s.objectField("content");
        const lines = lineCount(args.content);
        try s.write(std.fmt.bufPrint(&note, "{s}{d} line{s} written; read the file for its text]", .{ stub_marker, lines, if (lines == 1) "" else "s" }) catch unreachable);
    } else {
        const edits = if (args.edits) |e| switch (e) {
            .array => |a| a.items.len,
            else => 1,
        } else 1;
        try s.objectField("old_string");
        try s.write(std.fmt.bufPrint(&note, "{s}{d} edit{s}; read the file for its text]", .{ stub_marker, edits, if (edits == 1) "" else "s" }) catch unreachable);
        try s.objectField("new_string");
        try s.write("");
    }
    try s.endObject();
    return try out.toOwnedSlice();
}

fn lineCount(text: []const u8) usize {
    return std.mem.count(u8, text, "\n") + @intFromBool(text.len > 0 and text[text.len - 1] != '\n');
}

/// The hand-off message: the header, the ledger, the model's `summary`,
/// and, mid-turn, the `request` in progress. Caller owns the result.
pub fn write(alloc: Allocator, ledger: Ledger, summary: []const u8, request: []const u8) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    const w = &out.writer;
    try w.print("{s}\n\n## What the user asked\n", .{header});
    for (ledger.asks, 1..) |ask, n| {
        const kept = cut(ask, ask_bytes);
        try w.print("{d}. {s}{s}\n", .{ n, kept, if (kept.len < ask.len) " [cut]" else "" });
    }
    if (ledger.agreed.len > 0) {
        const kept = cut(ledger.agreed, agreed_bytes);
        try w.print("\n## The answer the user replied to\n{s}{s}\n", .{ kept, if (kept.len < ledger.agreed.len) " [cut]" else "" });
    }
    if (ledger.touched.len > 0) {
        try w.writeAll("\n## Files written or edited\n");
        for (ledger.touched) |t| try w.print("- {s}: {s}\n", .{ t.path, t.summary });
    }
    if (ledger.failing.len > 0) try w.print("\n## The last command that failed (its end)\n```\n{s}\n```\n", .{ledger.failing});
    try w.print("\n## Summary\n{s}\n", .{std.mem.trim(u8, summary, " \t\r\n")});
    if (request.len > 0) try w.print("\n## The request in progress\n{s}\n\nContinue it from the summary's next step.\n", .{request});
    return try out.toOwnedSlice();
}

/// Keeps `asks` within `max_asks`: the first few, then the latest.
/// Frees the one it drops.
pub fn pushAsk(alloc: Allocator, asks: *std.ArrayList([]u8), text: []const u8) !void {
    const owned = try alloc.dupe(u8, cut(text, ask_bytes));
    errdefer alloc.free(owned);
    if (asks.items.len == max_asks) alloc.free(asks.orderedRemove(first_asks));
    try asks.append(alloc, owned);
}

const testing = std.testing;

test "a long write or edit is shortened to its path and a stub; a short one, another tool, or a stub stays" {
    const alloc = testing.allocator;
    const content: [600]u8 = @splat('a');
    const write_args = "{\"path\":\"src/db.ts\",\"content\":\"" ++ content ++ "\\nb\\n\"}";
    const stub = (try argumentStub(alloc, "write_file", write_args)).?;
    defer alloc.free(stub);
    try testing.expectEqualStrings("{\"path\":\"src/db.ts\",\"content\":\"[shortened to fit the context: 2 lines written; read the file for its text]\"}", stub);
    try testing.expect(try argumentStub(alloc, "write_file", stub) == null);
    try testing.expect(try argumentStub(alloc, "write_file", "{\"path\":\"a\",\"content\":\"x\"}") == null);
    try testing.expect(try argumentStub(alloc, "bash", write_args) == null);
    const edit_args = "{\"path\":\"a.ts\",\"edits\":[{\"old_string\":\"" ++ content ++ "\",\"new_string\":\"y\"},{\"old_string\":\"z\",\"new_string\":\"w\"}]}";
    const edit_stub = (try argumentStub(alloc, "edit_file", edit_args)).?;
    defer alloc.free(edit_stub);
    try testing.expectEqualStrings("{\"path\":\"a.ts\",\"old_string\":\"[shortened to fit the context: 2 edits; read the file for its text]\",\"new_string\":\"\"}", edit_stub);
}

test "the hand-off message carries the ledger, the summary, and the request" {
    const alloc = testing.allocator;
    const text = try write(alloc, .{
        .asks = &.{ "build a todo app", "I confirm" },
        .agreed = "Plan: src/server.ts",
        .touched = &.{.{ .path = "src/db.ts", .summary = "Wrote 40 lines to src/db.ts" }},
        .failing = "error: x",
    }, "\nGoal: a todo app\n", "continue");
    defer alloc.free(text);
    try testing.expect(std.mem.startsWith(u8, text, header ++ "\n\n## What the user asked\n1. build a todo app\n2. I confirm\n"));
    try testing.expect(std.mem.indexOf(u8, text, "## The answer the user replied to\nPlan: src/server.ts\n") != null);
    try testing.expect(std.mem.indexOf(u8, text, "- src/db.ts: Wrote 40 lines to src/db.ts\n") != null);
    try testing.expect(std.mem.indexOf(u8, text, "## Summary\nGoal: a todo app\n") != null);
    try testing.expect(std.mem.endsWith(u8, text, "## The request in progress\ncontinue\n\nContinue it from the summary's next step.\n"));
}

test "asks keep the first few and the latest; cuts land on a code point" {
    const alloc = testing.allocator;
    var asks: std.ArrayList([]u8) = .empty;
    defer {
        for (asks.items) |a| alloc.free(a);
        asks.deinit(alloc);
    }
    for (0..max_asks + 3) |i| {
        var buf: [8]u8 = undefined;
        try pushAsk(alloc, &asks, std.fmt.bufPrint(&buf, "{d}", .{i}) catch unreachable);
    }
    try testing.expectEqual(@as(usize, max_asks), asks.items.len);
    try testing.expectEqualStrings("3", asks.items[first_asks - 1]);
    try testing.expectEqualStrings("7", asks.items[first_asks]);
    try testing.expectEqualStrings("18", asks.items[max_asks - 1]);
    try testing.expectEqualStrings("a", cut("aé", 2));
    try testing.expectEqualStrings("c\nd", tailLines("a\nb\nc\nd\n", 2));
}
