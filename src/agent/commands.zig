//! Slash commands and what the completion list offers.
//!
//! A line the user submits is a prompt unless it is unmistakably a command:
//! it starts with `/`, and its first word is nothing but ASCII letters. That
//! rule is what keeps `/usr/bin/env is fine` a question and `/ctx 16384` an
//! instruction, without a mode, a prefix key, or an escape (docs/spec.md § Editor).
//! A line that starts with `!` is a shell command: `!cmd` runs it and sends
//! its output to the model, `!!cmd` runs it for the user alone.
//!
//! Parsing is pure and lives here; executing belongs to the agent, which owns
//! the engine and the session. The table below is also the help text and the
//! completion list, so the three can never disagree.
const std = @import("std");

const Allocator = std.mem.Allocator;

/// The commands this phase implements. The table below is also the help text
/// and the completion list.
pub const Kind = enum { new, resume_session, ctx, think, save, image, help, shell };

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
    .{ .kind = .image, .name = "image", .argument = "<path>", .summary = "attach an image to the prompt (or drop the file onto the window)" },
    .{ .kind = .help, .name = "help", .summary = "keys and commands" },
};

/// The `!` form, described like a command but not in the table: it is not
/// a `/word`, so it is neither completed nor listed among them.
pub const shell_spec: Spec = .{ .kind = .shell, .name = "!", .argument = "<command>", .summary = "run a shell command and send its output; !! runs it without sending" };

/// A shell line: the command after the `!`, and whether its output is
/// sent to the model as the next message (`!`) or only shown (`!!`).
pub const Shell = struct { command: []const u8, send: bool };

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
    /// An image path to attach as a chip; the agent resolves and reads it.
    image: []const u8,
    help,
    shell: Shell,
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
    if (trimmed.len >= 1 and trimmed[0] == '!') {
        const send = !(trimmed.len >= 2 and trimmed[1] == '!');
        const command = std.mem.trim(u8, trimmed[if (send) 1 else 2..], " \t");
        if (command.len == 0) return .{ .usage = shell_spec };
        return .{ .command = .{ .shell = .{ .command = command, .send = send } } };
    }
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
            .image => if (rest.len == 0) .{ .usage = spec } else .{ .command = .{ .image = rest } },
            .shell => unreachable, // never in the table
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
        .{ "Ctrl-O", "cycle the tool rows of the last turn: summary, output, folded" },
        .{ "Ctrl-E", "expand a paste, file, or image chip into editable text" },
        .{ "Ctrl-G", "edit the input in $VISUAL or $EDITOR" },
        .{ "Ctrl-X", "copy the last answer to the clipboard" },
        .{ "Ctrl-T / Ctrl-W", "cycle reasoning effort / context window" },
        .{ "Ctrl-N", "new session" },
        .{ "Ctrl-C / Ctrl-D", "cancel a turn, or quit" },
    };
    for (keys) |pair| try rows.append(alloc, try alloc.print("    {s: <17}  {s}", .{ pair[0], pair[1] }));
    try rows.append(alloc, "");
    try rows.append(alloc, "  commands");
    for (table) |spec| {
        const name = if (spec.argument.len > 0)
            try alloc.print("/{s} {s}", .{ spec.name, spec.argument })
        else
            try alloc.print("/{s}", .{spec.name});
        try rows.append(alloc, try alloc.print("    {s: <17}  {s}", .{ name, spec.summary }));
    }
    try rows.append(alloc, try alloc.print("    {s: <17}  {s}", .{ "!<command>", shell_spec.summary }));
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
        try names.append(alloc, try alloc.print("{s}{s}{s}", .{ prefix[0..cut], entry.name, suffix }));
        if (names.items.len >= max_paths) break;
    }
    std.mem.sort([]const u8, names.items, {}, lessThan);
    return names.toOwnedSlice(alloc);
}

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

/// What an `@` token offers: a query with a `/` (or none) lists that
/// directory by prefix, so Tab walks into it; a bare word is matched
/// fuzzily against the whole tree, so `@faq` finds `docs/faq.md`.
pub fn completePaths(alloc: Allocator, io: std.Io, dir: std.Io.Dir, query: []const u8) ![]const []const u8 {
    if (query.len == 0 or std.mem.indexOfScalar(u8, query, '/') != null) return workspacePaths(alloc, io, dir, query);
    return fuzzyPaths(alloc, io, dir, query);
}

/// Entries the tree walk visits at most per completion: the list is rebuilt
/// on every key, so a large tree is cut rather than slowing the editor.
pub const max_walk = 20_000;

/// Directories never walked: version control and build output, which hold
/// nothing a prompt names and can hold most of a tree's entries.
const skipped_dirs = [_][]const u8{ ".git", "node_modules", "zig-out", ".zig-cache", "__pycache__" };

/// Workspace entries whose path matches `query` (`fuzzyScore`), best first,
/// at most `max_paths`, directories with their trailing separator. Hidden
/// entries take part only when the query starts with a dot.
pub fn fuzzyPaths(alloc: Allocator, io: std.Io, dir: std.Io.Dir, query: []const u8) ![]const []const u8 {
    const Hit = struct { path: []const u8, score: i32 };
    var hits: std.ArrayList(Hit) = .empty;
    defer hits.deinit(alloc);
    errdefer for (hits.items) |hit| alloc.free(hit.path);
    const hidden = query.len > 0 and query[0] == '.';
    var root = dir.openDir(io, ".", .{ .iterate = true }) catch return &.{};
    defer root.close(io);
    var walker = try root.walkSelectively(alloc);
    defer walker.deinit();
    var visited: usize = 0;
    while (visited < max_walk) : (visited += 1) {
        const entry = (walker.next(io) catch continue) orelse break;
        const name = entry.basename;
        if (name.len == 0) continue;
        if (entry.kind == .directory) {
            const skip = for (skipped_dirs) |s| {
                if (std.mem.eql(u8, name, s)) break true;
            } else false;
            if (skip or (name[0] == '.' and !hidden)) continue;
            walker.enter(io, entry) catch {};
        } else if (name[0] == '.' and !hidden) continue;
        const score = fuzzyScore(query, entry.path) orelse continue;
        const suffix: []const u8 = if (entry.kind == .directory) "/" else "";
        const path = try alloc.print("{s}{s}", .{ entry.path, suffix });
        hits.append(alloc, .{ .path = path, .score = score }) catch |err| {
            alloc.free(path);
            return err;
        };
    }
    std.mem.sort(Hit, hits.items, {}, struct {
        fn better(_: void, a: Hit, b: Hit) bool {
            if (a.score != b.score) return a.score > b.score;
            if (a.path.len != b.path.len) return a.path.len < b.path.len;
            return std.mem.lessThan(u8, a.path, b.path);
        }
    }.better);
    const kept = @min(hits.items.len, max_paths);
    for (hits.items[kept..]) |hit| alloc.free(hit.path);
    const out = try alloc.alloc([]const u8, kept);
    for (hits.items[0..kept], out) |hit, *o| o.* = hit.path;
    hits.clearRetainingCapacity();
    return out;
}

/// How well `query` matches `path` as an in-order, case-insensitive
/// subsequence, or null when it does not. Matched inside the last component
/// when it can be, else across the path; consecutive characters, a
/// character starting a word, and a basename that starts with the query
/// score higher, and a longer path slightly lower.
pub fn fuzzyScore(query: []const u8, path: []const u8) ?i32 {
    if (query.len == 0) return 0;
    const trimmed = std.mem.trimEnd(u8, path, "/");
    const base = if (std.mem.lastIndexOfScalar(u8, trimmed, '/')) |i| i + 1 else 0;
    var score: i32 = undefined;
    if (subsequence(query, trimmed, base)) |s| {
        score = s + 10;
        if (std.ascii.startsWithIgnoreCase(trimmed[base..], query)) score += 20;
    } else score = subsequence(query, trimmed, 0) orelse return null;
    return score - @as(i32, @intCast(@min(trimmed.len, 400) / 8));
}

fn subsequence(query: []const u8, text: []const u8, from: usize) ?i32 {
    var score: i32 = 0;
    var at = from;
    var previous: ?usize = null;
    for (query) |c| {
        const lower = std.ascii.toLower(c);
        while (at < text.len and std.ascii.toLower(text[at]) != lower) at += 1;
        if (at == text.len) return null;
        score += 1;
        if (previous != null and previous.? + 1 == at) score += 5;
        if (at == 0 or std.mem.indexOfScalar(u8, "/_-. ", text[at - 1]) != null) score += 8;
        previous = at;
        at += 1;
    }
    return score;
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
    try testing.expectEqualStrings("shots/a.png", parse("/image shots/a.png").?.command.image);
    try testing.expectEqual(Kind.image, parse("/image").?.usage.kind);
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

test "a line that starts with ! is a shell command, sent or shown" {
    const sent = parse("!ls -la").?.command.shell;
    try testing.expectEqualStrings("ls -la", sent.command);
    try testing.expect(sent.send);
    const shown = parse("!! make check ").?.command.shell;
    try testing.expectEqualStrings("make check", shown.command);
    try testing.expect(!shown.send);
    // Leading space is trimmed on both forms; the marker is not a command.
    try testing.expectEqualStrings("git status", parse("!  git status").?.command.shell.command);
    try testing.expectEqual(Kind.shell, parse("!").?.usage.kind);
    try testing.expectEqual(Kind.shell, parse("!!").?.usage.kind);
    try testing.expectEqual(Kind.shell, parse("!!   ").?.usage.kind);
    // Only at the start: an exclamation elsewhere is prose.
    try testing.expect(parse("what!") == null);
    try testing.expect(parse("really? ! yes") == null);
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
    try testing.expect(std.mem.indexOf(u8, joined.items, "!<command>") != null);
    try testing.expect(std.mem.indexOf(u8, joined.items, "Ctrl-O") != null);
    try testing.expect(std.mem.indexOf(u8, joined.items, "Ctrl-G") != null);
    try testing.expect(std.mem.indexOf(u8, joined.items, "Ctrl-X") != null);
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

test "a bare @ word matches fuzzily across the tree, best first" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const io = testing.io;
    const alloc = testing.allocator;
    for ([_][]const u8{ "docs/faq.md", "docs/design.md", "src/shapes/polygon.py", "tests/test_polygon.py", "README.md", ".git/config", "node_modules/faq/index.js" }) |path| {
        if (std.fs.path.dirname(path)) |parent| try tmp.dir.createDirPath(io, parent);
        try tmp.dir.writeFile(io, .{ .sub_path = path, .data = "" });
    }

    const faq = try completePaths(alloc, io, tmp.dir, "faq");
    defer free(alloc, faq);
    // Skipped directories are not walked: the one match is the document.
    try testing.expectEqual(@as(usize, 1), faq.len);
    try testing.expectEqualStrings("docs/faq.md", faq[0]);

    // A basename match outranks one spread across directories.
    const poly = try completePaths(alloc, io, tmp.dir, "poly");
    defer free(alloc, poly);
    try testing.expectEqualStrings("src/shapes/polygon.py", poly[0]);
    try testing.expectEqualStrings("tests/test_polygon.py", poly[1]);

    // Out of order is no match; case does not matter.
    const none = try completePaths(alloc, io, tmp.dir, "qaf");
    defer free(alloc, none);
    try testing.expectEqual(@as(usize, 0), none.len);
    const upper = try completePaths(alloc, io, tmp.dir, "DESIGN");
    defer free(alloc, upper);
    try testing.expectEqualStrings("docs/design.md", upper[0]);

    // Directories match too, and a slash keeps the walk-into behaviour.
    const docs = try completePaths(alloc, io, tmp.dir, "docs");
    defer free(alloc, docs);
    try testing.expectEqualStrings("docs/", docs[0]);
    const inside = try completePaths(alloc, io, tmp.dir, "docs/f");
    defer free(alloc, inside);
    try testing.expectEqual(@as(usize, 1), inside.len);
    try testing.expectEqualStrings("docs/faq.md", inside[0]);
}

test "fuzzy scores prefer word starts and consecutive characters" {
    try testing.expect(fuzzyScore("faq", "docs/faq.md").? > fuzzyScore("faq", "docs/fix_a_queue.md").?);
    try testing.expect(fuzzyScore("rd", "README.md").? > fuzzyScore("rd", "src/bird.py").?);
    try testing.expect(fuzzyScore("xyz", "docs/faq.md") == null);
    try testing.expectEqual(@as(?i32, 0), fuzzyScore("", "anything"));
}

fn free(alloc: Allocator, items: []const []const u8) void {
    for (items) |item| alloc.free(item);
    alloc.free(items);
}
