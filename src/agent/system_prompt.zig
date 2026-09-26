//! The agent's system prompt, built from sections so one can change without
//! re-priming the rest by hand: identity and workspace, the working rules,
//! one guideline per tool, the cost rule, the environment (date, the user's
//! own shell commands), and the project's instructions file. The profile
//! renders the tool definitions themselves; this text is what follows them.
//!
//! Every sentence here was measured on the playground task list
//! (`scripts/agent-eval.py`, docs/development.md § The agent's task list):
//! change the text, run the list, compare. `--system-prompt <file>` replaces
//! the built sections with a file's text for exactly that purpose; the
//! project instructions still follow it.
const std = @import("std");
const tools = @import("tools/root.zig");

const Allocator = std.mem.Allocator;

/// The most of an instructions file the prompt carries. A larger file is cut
/// at a line boundary with a marked cut: the window is small and the file is
/// primed into every session.
pub const max_instructions_bytes: usize = 8192;

/// `agent.instructions` values with a meaning of their own; any other value
/// is a workspace-relative path.
pub const instructions_auto = "auto";
pub const instructions_off = "off";
/// What `auto` looks for, in order.
pub const default_files = [_][]const u8{ "AGENTS.md", "CLAUDE.md" };

/// A project instructions file as read from the workspace. Owned.
pub const Instructions = struct {
    /// The name the prompt tags the section with (the path as configured).
    name: []u8,
    /// The file's text, cut to `max_instructions_bytes` with a marker.
    text: []u8,
    cut: bool,

    pub fn deinit(self: *Instructions, alloc: Allocator) void {
        alloc.free(self.name);
        alloc.free(self.text);
        self.* = undefined;
    }
};

pub const Options = struct {
    /// The workspace root, canonical.
    root: []const u8,
    /// `YYYY-MM-DD`, or null to leave the date out (tests).
    date: ?[]const u8 = null,
    instructions: ?Instructions = null,
    /// A file's text standing in for every built section (`--system-prompt`);
    /// the instructions section still follows it.
    override: ?[]const u8 = null,
};

/// Reads the instructions file `setting` names (`auto`, `off`, or a path)
/// from the workspace. Null when there is none: `off`, no default file
/// present, or the named file missing. An unreadable named file is the
/// caller's error; an unreadable default file is skipped.
pub fn load(alloc: Allocator, io: std.Io, dir: std.Io.Dir, setting: []const u8) !?Instructions {
    if (std.mem.eql(u8, setting, instructions_off)) return null;
    if (std.mem.eql(u8, setting, instructions_auto)) {
        for (default_files) |name| {
            if (read(alloc, io, dir, name) catch null) |found| return found;
        }
        return null;
    }
    return read(alloc, io, dir, setting) catch |err| switch (err) {
        error.FileNotFound => null,
        else => err,
    };
}

fn read(alloc: Allocator, io: std.Io, dir: std.Io.Dir, name: []const u8) !Instructions {
    const file = try dir.openFile(io, name, .{});
    defer file.close(io);
    var buffer: [max_instructions_bytes + 1]u8 = undefined;
    var reader = file.reader(io, &.{});
    const len = try reader.interface.readSliceShort(&buffer);
    const owned_name = try alloc.dupe(u8, name);
    errdefer alloc.free(owned_name);
    const text = try clip(alloc, buffer[0..len]);
    return .{ .name = owned_name, .text = text, .cut = len > max_instructions_bytes };
}

/// The text up to the cap, cut at its last line boundary and marked; the
/// whole text otherwise. Owned by the caller.
pub fn clip(alloc: Allocator, text: []const u8) ![]u8 {
    const trimmed = std.mem.trimEnd(u8, text, " \t\r\n");
    if (text.len <= max_instructions_bytes) return alloc.dupe(u8, trimmed);
    const kept_end = std.mem.lastIndexOfScalar(u8, text[0..max_instructions_bytes], '\n') orelse max_instructions_bytes;
    const kept = std.mem.trimEnd(u8, text[0..kept_end], " \t\r\n");
    return std.fmt.allocPrint(alloc, "{s}\n[cut here: only the first {d} bytes of the file are shown]", .{ kept, kept.len });
}

/// The one guideline rendered under a tool's name, keyed by the registry
/// name; null for a tool without one.
pub fn guideline(name: []const u8) ?[]const u8 {
    const table = [_]struct { name: []const u8, text: []const u8 }{
        .{ .name = "read_file", .text = "read a file with this, not with cat; ask for the region you need. When it refuses a file (too large, not text), use head, wc, or grep in bash instead of smaller reads." },
        .{ .name = "grep", .text = "the pattern is a literal string, never a regular expression; one word per call. When a question spans many files, search before you read." },
        .{ .name = "write_file", .text = "only for a new file or a full rewrite, the whole content at once and ending with a newline; an existing file is changed with edit_file." },
        .{ .name = "edit_file", .text = "keep old_string as short as it can be while still unique; when the result says it matched more than once, add a line of context, never rewrite the file. The result shows the diff: do not read the file again to check it." },
        .{ .name = "bash", .text = "one command at a time, run from the workspace (no cd). Run tests the way the project does (its Makefile, or the runner its tests import); a failed command's exit status goes in your answer." },
    };
    for (table) |entry| if (std.mem.eql(u8, entry.name, name)) return entry.text;
    return null;
}

pub const identity = "You are nuclis, a coding agent working in {s}.";

pub const rules =
    "Complete the user's task with the available tools, and nothing beyond it: an unrelated bug or improvement " ++
    "you notice is one sentence in your answer, not an edit. Read a file before you change it, " ++
    "make one change at a time, and say what you did when you are finished. " ++
    "Never invent a tool result: wait for the output before you continue. " ++
    "Your context is small and a long file arrives in pages: for a question about a whole file " ++
    "prefer one shell command (grep, sort, awk, wc) over reading it page by page, let the shell count (wc -l, grep -c) rather than counting lines yourself, and say when you saw only part of it. " ++
    "Asked to show a file longer than one page, quote the page you received in your answer (the user does not see tool output unless you quote it), name the lines it covers, and offer the rest; answer any question about the whole file with one command. " ++
    "Quote file contents only from tool output you received in this turn; never reconstruct a file from memory. " ++
    "When a request needs more than the context can hold, say so and offer a summary or a command instead.";

pub const cost_rule =
    "Every line you write costs the user about a second: answer with what was asked and nothing decorative. " ++
    "When the task is done, stop; do not re-read or re-run what the tool output already showed.";

pub const shell_rule = "A user message that starts with \"$ \" is a command the user ran in their own shell, followed by its output.";

/// The whole system text. Owned by the caller.
pub fn build(alloc: Allocator, options: Options) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(alloc);
    errdefer out.deinit();
    const w = &out.writer;
    if (options.override) |text| {
        try w.writeAll(std.mem.trimEnd(u8, text, " \t\r\n"));
        try w.writeByte('\n');
    } else {
        try w.print(identity ++ "\n\n", .{options.root});
        try w.writeAll(rules ++ "\n\n");
        try w.writeAll("Tool guidelines:\n");
        for (tools.all) |tool| {
            if (guideline(tool.name)) |text| try w.print("- {s}: {s}\n", .{ tool.name, text });
        }
        try w.writeAll("\n" ++ cost_rule ++ "\n\n");
        if (options.date) |date| try w.print("Today is {s}. ", .{date});
        try w.writeAll(shell_rule ++ "\n");
    }
    if (options.instructions) |ins| {
        try w.print("\n<project_instructions file=\"{s}\">\n{s}\n</project_instructions>\n", .{ ins.name, ins.text });
    }
    return out.toOwnedSlice();
}

// ----- tests -----

const testing = std.testing;

test "the built text is the measured one" {
    // The task list measured exactly this text (docs/development.md § The
    // agent's task list); a change here is re-measured, then re-pinned.
    const alloc = testing.allocator;
    const text = try build(alloc, .{ .root = "/w", .date = "2026-09-22" });
    defer alloc.free(text);
    try testing.expectEqualStrings(@embedFile("fixtures/system_prompt.txt"), text);
}

test "the sections come in order and every tool with a guideline is listed under it" {
    const alloc = testing.allocator;
    const text = try build(alloc, .{ .root = "/w", .date = "2026-09-22" });
    defer alloc.free(text);
    const marks = [_][]const u8{ "working in /w.", "Complete the user's task", "Tool guidelines:\n- read_file:", "- grep:", "- write_file:", "- edit_file:", "- bash:", "costs the user about a second", "Today is 2026-09-22. A user message" };
    var at: usize = 0;
    for (marks) |mark| {
        const found = std.mem.indexOfPos(u8, text, at, mark) orelse return error.TestUnexpectedResult;
        at = found + mark.len;
    }
    // Only the tools with a guideline are listed; the table names none twice.
    try testing.expectEqual(@as(usize, 5), std.mem.count(u8, text, "\n- "));
    try testing.expect(std.mem.indexOf(u8, text, "project_instructions") == null);
}

test "the instructions section follows, tagged with the file's name; an override replaces the rest" {
    const alloc = testing.allocator;
    const ins: Instructions = .{ .name = try alloc.dupe(u8, "AGENTS.md"), .text = try alloc.dupe(u8, "Keep it pure."), .cut = false };
    var owned = ins;
    defer owned.deinit(alloc);
    const text = try build(alloc, .{ .root = "/w", .instructions = ins });
    defer alloc.free(text);
    try testing.expect(std.mem.endsWith(u8, text, "\n<project_instructions file=\"AGENTS.md\">\nKeep it pure.\n</project_instructions>\n"));
    try testing.expect(std.mem.indexOf(u8, text, "Tool guidelines:") != null);

    const replaced = try build(alloc, .{ .root = "/w", .instructions = ins, .override = "Be terse.\n\n" });
    defer alloc.free(replaced);
    try testing.expectEqualStrings("Be terse.\n\n<project_instructions file=\"AGENTS.md\">\nKeep it pure.\n</project_instructions>\n", replaced);
}

test "clip keeps a short text whole and cuts a long one at a line boundary with a marker" {
    const alloc = testing.allocator;
    const short = try clip(alloc, "one\ntwo\n\n");
    defer alloc.free(short);
    try testing.expectEqualStrings("one\ntwo", short);

    var long: std.ArrayList(u8) = .empty;
    defer long.deinit(alloc);
    while (long.items.len <= max_instructions_bytes + 100) try long.appendSlice(alloc, "a line of the file that goes on\n");
    const cut = try clip(alloc, long.items);
    defer alloc.free(cut);
    try testing.expect(cut.len < max_instructions_bytes + 80);
    try testing.expect(std.mem.endsWith(u8, cut, "bytes of the file are shown]"));
    // The cut lands on a line boundary: the kept part ends with a whole line.
    const marker = std.mem.indexOf(u8, cut, "\n[cut here").?;
    try testing.expect(std.mem.endsWith(u8, cut[0..marker], "goes on"));
}

test "load: off is none, auto takes AGENTS.md then CLAUDE.md, a named file is read, a missing one is none" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try testing.expect((try load(alloc, testing.io, tmp.dir, instructions_off)) == null);
    try testing.expect((try load(alloc, testing.io, tmp.dir, instructions_auto)) == null);
    try testing.expect((try load(alloc, testing.io, tmp.dir, "missing.md")) == null);

    try tmp.dir.writeFile(testing.io, .{ .sub_path = "CLAUDE.md", .data = "claude rules\n" });
    var fallback = (try load(alloc, testing.io, tmp.dir, instructions_auto)).?;
    defer fallback.deinit(alloc);
    try testing.expectEqualStrings("CLAUDE.md", fallback.name);
    try testing.expectEqualStrings("claude rules", fallback.text);

    try tmp.dir.writeFile(testing.io, .{ .sub_path = "AGENTS.md", .data = "agents rules\n" });
    var first = (try load(alloc, testing.io, tmp.dir, instructions_auto)).?;
    defer first.deinit(alloc);
    try testing.expectEqualStrings("AGENTS.md", first.name);
    try testing.expect(!first.cut);

    try tmp.dir.createDirPath(testing.io, "docs");
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "docs/x.md", .data = "named\n" });
    var named = (try load(alloc, testing.io, tmp.dir, "docs/x.md")).?;
    defer named.deinit(alloc);
    try testing.expectEqualStrings("docs/x.md", named.name);
    try testing.expectEqualStrings("named", named.text);
}
