//! Slash commands and what the completion list offers.
//!
//! A line the user submits is a prompt unless it is unmistakably a command:
//! it starts with `/`, and its first word is nothing but ASCII letters. That
//! rule is what keeps `/usr/bin/env is fine` a question and `/ctx 16384` an
//! instruction, without a mode, a prefix key, or an escape (agent-spec § Keys
//! and commands).
//!
//! Parsing is pure and lives here; executing belongs to the agent, which owns
//! the engine and the session. The table below is also the help text and the
//! completion list, so the three can never disagree.
const std = @import("std");

const Allocator = std.mem.Allocator;

/// The commands this phase implements. The table below is also the help text
/// and the completion list.
pub const Kind = enum { new, resume_session, ctx, think, save, help };

pub const Spec = struct {
    kind: Kind,
    name: []const u8,
    /// How the argument is written in help, empty when there is none.
    argument: []const u8 = "",
    summary: []const u8,
};

pub const table = [_]Spec{
    .{ .kind = .new, .name = "new", .summary = "start a new session (as Ctrl-N)" },
    .{ .kind = .resume_session, .name = "resume", .summary = "pick a saved session and replay it" },
    .{ .kind = .ctx, .name = "ctx", .argument = "<n>", .summary = "context window in tokens (as Ctrl-W, with a value)" },
    .{ .kind = .think, .name = "think", .argument = "<effort>", .summary = "reasoning effort: off, low, medium, high, xhigh (as Ctrl-T)" },
    .{ .kind = .save, .name = "save", .argument = "[path]", .summary = "export this session as markdown" },
    .{ .kind = .help, .name = "help", .summary = "keys and commands" },
};

pub const Command = union(enum) {
    new,
    /// Open the session picker; the choice is made interactively.
    resume_session,
    /// Validated as a number here; the range is the configuration's business.
    ctx: usize,
    /// The effort as written; the agent maps it onto the profile's enum.
    think: []const u8,
    /// A path, or null for the default under `~/.nuclis/agent/exports/`.
    save: ?[]const u8,
    help,
};

pub const Result = union(enum) {
    command: Command,
    /// A `/word` that names nothing.
    unknown: []const u8,
    /// A known command whose argument is missing or unreadable.
    usage: Spec,
};

/// Parses a submitted line. Null means "this is a prompt, not a command".
pub fn parse(line: []const u8) ?Result {
    const trimmed = std.mem.trim(u8, line, " \t\r\n");
    if (trimmed.len < 2 or trimmed[0] != '/') return null;
    const word_end = std.mem.indexOfAny(u8, trimmed, " \t") orelse trimmed.len;
    const word = trimmed[1..word_end];
    // A path, a fraction, or a date is not a command attempt.
    for (word) |c| if (!std.ascii.isAlphabetic(c)) return null;
    if (word.len > 16) return null;
    const rest = std.mem.trim(u8, trimmed[word_end..], " \t");
    for (table) |spec| {
        if (!std.ascii.eqlIgnoreCase(spec.name, word)) continue;
        return switch (spec.kind) {
            .new => .{ .command = .new },
            .resume_session => .{ .command = .resume_session },
            .help => .{ .command = .help },
            .ctx => if (std.fmt.parseInt(usize, rest, 10) catch null) |value|
                .{ .command = .{ .ctx = value } }
            else
                .{ .usage = spec },
            .think => if (rest.len == 0) .{ .usage = spec } else .{ .command = .{ .think = rest } },
            .save => .{ .command = .{ .save = if (rest.len == 0) null else rest } },
        };
    }
    return .{ .unknown = word };
}

/// The commands whose names start with `prefix` (given without the slash).
/// Writes into `out` and returns the used part, so the caller needs no
/// allocation for a list this small.
pub fn matching(prefix: []const u8, out: *[table.len]Spec) []const Spec {
    var count: usize = 0;
    for (table) |spec| {
        if (!std.mem.startsWith(u8, spec.name, prefix)) continue;
        out[count] = spec;
        count += 1;
    }
    return out[0..count];
}

/// The help text, as the rows an `info` block renders: a title, the keys,
/// then the command table above, so the two halves of the interface are
/// described in one place. Every command and key is spelled once; the same
/// `table` drives completion, so help and completion cannot disagree. Caller
/// owns the strings (an arena, in the agent).
pub fn help(alloc: Allocator, ascii: bool) ![]const []const u8 {
    var rows: std.ArrayList([]const u8) = .empty;
    try rows.append(alloc, "");
    try rows.append(alloc, if (ascii) "nuclis agent - keys and commands" else "nuclis agent — keys and commands");
    try rows.append(alloc, "");
    try rows.append(alloc, "  keys");
    const keys = [_][2][]const u8{
        .{ "Enter", "send; while a turn runs, queue the message for the next one" },
        .{ "Shift-Enter", "newline (Ctrl-J too)" },
        .{ "Up / Down", "move in the input; history at the first and last row" },
        .{ "Tab", "complete a /command or an @path, otherwise fold thinking" },
        .{ "Ctrl-E", "expand a paste chip into editable text" },
        .{ "Ctrl-T / Ctrl-W", "cycle reasoning effort / context window" },
        .{ "Ctrl-N", "new session" },
        .{ "Ctrl-C / Ctrl-D", "cancel a turn, or quit" },
    };
    for (keys) |pair| try rows.append(alloc, try std.fmt.allocPrint(alloc, "    {s: <17}  {s}", .{ pair[0], pair[1] }));
    try rows.append(alloc, "");
    try rows.append(alloc, "  commands");
    for (table) |spec| {
        const name = if (spec.argument.len > 0)
            try std.fmt.allocPrint(alloc, "/{s} {s}", .{ spec.name, spec.argument })
        else
            try std.fmt.allocPrint(alloc, "/{s}", .{spec.name});
        try rows.append(alloc, try std.fmt.allocPrint(alloc, "    {s: <17}  {s}", .{ name, spec.summary }));
    }
    return rows.toOwnedSlice(alloc);
}

// ----- path completion -----

/// Entries offered for an `@prefix` token, workspace-relative. Directories
/// keep their trailing separator so the next Tab continues inside them.
/// Hidden entries are offered only when the prefix asks for them, and the
/// count is bounded: a completion list is not a directory listing.
pub const max_paths = 50;

pub fn workspacePaths(alloc: Allocator, io: std.Io, dir: std.Io.Dir, prefix: []const u8) ![]const []const u8 {
    var names: std.ArrayList([]const u8) = .empty;
    errdefer names.deinit(alloc);
    const cut = if (std.mem.lastIndexOfScalar(u8, prefix, '/')) |i| i + 1 else 0;
    const parent = if (cut == 0) "." else prefix[0 .. cut - 1];
    const base = prefix[cut..];
    var opened = dir.openDir(io, if (parent.len == 0) "/" else parent, .{ .iterate = true }) catch return names.toOwnedSlice(alloc);
    defer opened.close(io);
    var iterator = opened.iterate();
    while (try iterator.next(io)) |entry| {
        if (entry.name.len == 0) continue;
        if (entry.name[0] == '.' and !(base.len > 0 and base[0] == '.')) continue;
        if (!std.mem.startsWith(u8, entry.name, base)) continue;
        const suffix: []const u8 = if (entry.kind == .directory) "/" else "";
        try names.append(alloc, try std.fmt.allocPrint(alloc, "{s}{s}{s}", .{ prefix[0..cut], entry.name, suffix }));
        if (names.items.len >= max_paths) break;
    }
    std.mem.sort([]const u8, names.items, {}, lessThan);
    return names.toOwnedSlice(alloc);
}

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

// ----- tests -----

const testing = std.testing;

test "a line is a command only when it is unmistakably one" {
    try testing.expectEqual(Command.new, parse("/new").?.command);
    try testing.expectEqual(Command.resume_session, parse("/resume").?.command);
    try testing.expectEqual(Command.help, parse("  /help  ").?.command);
    try testing.expectEqual(@as(usize, 16384), parse("/ctx 16384").?.command.ctx);
    try testing.expectEqualStrings("medium", parse("/think medium").?.command.think);
    try testing.expect(parse("/save").?.command.save == null);
    try testing.expectEqualStrings("out.md", parse("/save out.md").?.command.save.?);
    // Anything else is a prompt, including the paths that start with a slash.
    try testing.expect(parse("/usr/bin/env is fine") == null);
    try testing.expect(parse("/2 of them") == null);
    try testing.expect(parse("what is 1/2?") == null);
    try testing.expect(parse("/") == null);
    try testing.expect(parse("") == null);
    // A `/word` that names nothing is reported rather than sent to the model.
    try testing.expectEqualStrings("nope", parse("/nope").?.unknown);
    // A known command with an unusable argument asks for the right one.
    try testing.expectEqual(Kind.ctx, parse("/ctx lots").?.usage.kind);
    try testing.expectEqual(Kind.ctx, parse("/ctx").?.usage.kind);
    try testing.expectEqual(Kind.think, parse("/think").?.usage.kind);
}

test "completion offers the commands that start with what was typed" {
    var buffer: [table.len]Spec = undefined;
    try testing.expectEqual(table.len, matching("", &buffer).len);
    const n = matching("n", &buffer);
    try testing.expectEqual(@as(usize, 1), n.len);
    try testing.expectEqualStrings("new", n[0].name);
    try testing.expectEqual(@as(usize, 0), matching("zzz", &buffer).len);
}

test "the help text names every key and every command once" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const rows = try help(arena.allocator(), false);
    var joined: std.ArrayList(u8) = .empty;
    for (rows) |row| {
        try joined.appendSlice(arena.allocator(), row);
        try joined.append(arena.allocator(), '\n');
    }
    for (table) |spec| {
        var needle: [32]u8 = undefined;
        try testing.expect(std.mem.indexOf(u8, joined.items, try std.fmt.bufPrint(&needle, "/{s}", .{spec.name})) != null);
    }
    try testing.expect(std.mem.indexOf(u8, joined.items, "Shift-Enter") != null);
    try testing.expect(std.mem.indexOf(u8, joined.items, "queue the message") != null);
}

test "path completion is workspace-relative, bounded, and hides dotfiles unless asked" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = testing.io;
    const alloc = testing.allocator;
    try tmp.dir.writeFile(io, .{ .sub_path = "alpha.zig", .data = "" });
    try tmp.dir.writeFile(io, .{ .sub_path = "alphabet.txt", .data = "" });
    try tmp.dir.writeFile(io, .{ .sub_path = "beta.zig", .data = "" });
    try tmp.dir.writeFile(io, .{ .sub_path = ".hidden", .data = "" });
    try tmp.dir.createDirPath(io, "sub");
    try tmp.dir.writeFile(io, .{ .sub_path = "sub/inner.zig", .data = "" });

    const all = try workspacePaths(alloc, io, tmp.dir, "");
    defer free(alloc, all);
    try testing.expectEqual(@as(usize, 4), all.len); // the dotfile is not offered
    try testing.expectEqualStrings("alpha.zig", all[0]);
    try testing.expectEqualStrings("sub/", all[3]);

    const alphas = try workspacePaths(alloc, io, tmp.dir, "alpha");
    defer free(alloc, alphas);
    try testing.expectEqual(@as(usize, 2), alphas.len);

    const inside = try workspacePaths(alloc, io, tmp.dir, "sub/");
    defer free(alloc, inside);
    try testing.expectEqual(@as(usize, 1), inside.len);
    try testing.expectEqualStrings("sub/inner.zig", inside[0]);

    const hidden = try workspacePaths(alloc, io, tmp.dir, ".");
    defer free(alloc, hidden);
    try testing.expectEqual(@as(usize, 1), hidden.len);
    try testing.expectEqualStrings(".hidden", hidden[0]);

    // A directory that is not there offers nothing rather than failing.
    const missing = try workspacePaths(alloc, io, tmp.dir, "nowhere/x");
    defer free(alloc, missing);
    try testing.expectEqual(@as(usize, 0), missing.len);
}

fn free(alloc: Allocator, items: []const []const u8) void {
    for (items) |item| alloc.free(item);
    alloc.free(items);
}
