//! Resuming a stored conversation: list the sessions of a working directory,
//! find one by id, rebuild the message list the loop renders, and replay the
//! entries into a transcript.
//!
//! A resume is a **replay through the profile into a fresh session**, never a
//! state restore (docs/spec.md § Sessions and storage). Nothing here touches the
//! engine: the caller resets the session and hands the messages to
//! `loop.Agent.restore`. Strings in `message` borrow the `Loaded` arena, so
//! the caller keeps the load alive until the restore call returns.
const std = @import("std");
const inference = @import("inference");
const paths = @import("../paths.zig");
const tui = @import("../tui/root.zig");
const session = @import("session.zig");
const cache = @import("../completer/cache.zig");
const tools = @import("tools/root.zig");
const style = @import("../tui/style.zig");

const Allocator = std.mem.Allocator;
const Profile = inference.profiles;
const transcript = tui.transcript;
const screen = tui.screen;

/// One saved conversation, as a picker sees it: the header's facts plus the
/// path to load. Owned strings.
/// The id `find` resolves to the newest session of the workspace, for
/// `--resume` with no id. A real id is 32 hex digits, so the word cannot
/// collide.
pub const latest = "latest";

pub const Summary = struct {
    id: []u8,
    time: []u8,
    cwd: []u8,
    effort: []u8,
    ctx_size: usize,
    path: []u8,
    /// The first prompt's first line, at most `prompt_preview` bytes; empty
    /// when nothing was said.
    first_prompt: []u8,

    pub fn deinit(self: Summary, alloc: Allocator) void {
        alloc.free(self.id);
        alloc.free(self.time);
        alloc.free(self.cwd);
        alloc.free(self.effort);
        alloc.free(self.path);
        alloc.free(self.first_prompt);
    }
};

const prompt_preview: usize = 72;

/// The workspace's sessions as `nuclis agent ls` prints them: one form for
/// the terminal, the same fields as JSON.
pub const Listing = struct {
    schema_version: u32 = 1,
    cwd: []const u8,
    sessions: []const Summary,

    pub fn render(self: Listing, out: *std.Io.Writer, json: bool, sty: style.Style) !void {
        if (json) {
            try std.json.Stringify.value(self, .{ .whitespace = .indent_2 }, out);
            return out.writeByte('\n');
        }
        const off = sty.off();
        if (self.sessions.len == 0) {
            try out.print("{s}no saved sessions for{s} {s}{s}{s}\n", .{ sty.on(.label), off, sty.on(.code), self.cwd, off });
            return;
        }
        try out.print("{s}sessions for{s} {s}{s}{s} {s}(newest first; `nuclis agent --resume [<id>]` continues the newest or the named one){s}\n", .{ sty.on(.label), off, sty.on(.code), self.cwd, off, sty.on(.dim), off });
        for (self.sessions) |s| {
            const short = if (s.id.len > 8) s.id[0..8] else s.id;
            try out.print("  {s}{s}{s}  {s}  {s}{s:<6}{s} ctx {s}{d:<5}{s}", .{ sty.on(.keyword), short, off, s.time, sty.on(.number), s.effort, off, sty.on(.number), s.ctx_size, off });
            if (s.first_prompt.len > 0) try out.print("  {s}{s}{s}", .{ sty.on(.dim), s.first_prompt, off });
            try out.writeByte('\n');
        }
    }
};

/// `nuclis agent ls`: the sessions recorded for `cwd`.
pub fn ls(alloc: Allocator, io: std.Io, root_dir: []const u8, cwd: []const u8, json: bool, out: *std.Io.Writer, sty: style.Style) !void {
    const sessions = try list(alloc, io, root_dir, cwd);
    defer freeList(alloc, sessions);
    const listing: Listing = .{ .cwd = cwd, .sessions = sessions };
    try listing.render(out, json, sty);
}

pub fn freeList(alloc: Allocator, items: []Summary) void {
    for (items) |item| item.deinit(alloc);
    alloc.free(items);
}

/// The sessions recorded for `cwd`, newest first. A root with no session
/// directory yet is an empty list, not an error.
pub fn list(alloc: Allocator, io: std.Io, root_dir: []const u8, cwd: []const u8) ![]Summary {
    const dir_path = try sessionsPath(alloc, root_dir, cwd);
    defer alloc.free(dir_path);
    var dir = std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return &.{},
        else => return err,
    };
    defer dir.close(io);
    var out: std.ArrayList(Summary) = .empty;
    errdefer {
        for (out.items) |item| item.deinit(alloc);
        out.deinit(alloc);
    }
    var iterator = dir.iterate();
    while (try iterator.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".jsonl")) continue;
        const bytes = dir.readFileAlloc(io, entry.name, alloc, .limited(64 * 1024)) catch continue;
        defer alloc.free(bytes);
        const header = session.parseHeader(alloc, bytes) catch continue;
        defer header.deinit(alloc);
        const path = try std.fs.path.join(alloc, &.{ dir_path, entry.name });
        errdefer alloc.free(path);
        const first_prompt = try firstPrompt(alloc, bytes);
        errdefer alloc.free(first_prompt);
        try out.append(alloc, .{
            .id = try alloc.dupe(u8, header.id),
            .time = try alloc.dupe(u8, header.time),
            .cwd = try alloc.dupe(u8, header.cwd),
            .effort = try alloc.dupe(u8, header.effort),
            .ctx_size = header.ctx_size,
            .path = path,
            .first_prompt = first_prompt,
        });
    }
    std.mem.sort(Summary, out.items, {}, newerFirst);
    return out.toOwnedSlice(alloc);
}

/// The first line of the first user entry, cut to the preview length at a
/// code point; empty when the second line is not a user entry.
fn firstPrompt(alloc: Allocator, bytes: []const u8) ![]u8 {
    const start = (std.mem.indexOfScalar(u8, bytes, '\n') orelse return alloc.dupe(u8, "")) + 1;
    const end = std.mem.indexOfScalarPos(u8, bytes, start, '\n') orelse bytes.len;
    const parsed = std.json.parseFromSlice(struct { type: []const u8, text: []const u8 = "" }, alloc, bytes[start..end], .{ .ignore_unknown_fields = true }) catch return alloc.dupe(u8, "");
    defer parsed.deinit();
    if (!std.mem.eql(u8, parsed.value.type, "user")) return alloc.dupe(u8, "");
    var line = parsed.value.text;
    if (std.mem.indexOfScalar(u8, line, '\n')) |nl| line = line[0..nl];
    line = std.mem.trim(u8, line, " \t\r");
    if (line.len > prompt_preview) {
        var cut = prompt_preview;
        while (cut > 0 and (line[cut] & 0xC0) == 0x80) cut -= 1;
        return alloc.print("{s}…", .{line[0..cut]});
    }
    return alloc.dupe(u8, line);
}

fn newerFirst(_: void, a: Summary, b: Summary) bool {
    // RFC 3339 UTC at second resolution sorts lexically.
    return std.mem.lessThan(u8, b.time, a.time);
}

/// The path of the session named by `id`, or null when there is none. The
/// session file is `<stamp>_<id>.jsonl`, so the id alone locates it; an
/// argument that is a path (it names a `.jsonl` file or contains a separator)
/// is used as given, which is how print mode scripts a specific file.
pub fn find(alloc: Allocator, io: std.Io, root_dir: []const u8, cwd: []const u8, id: []const u8) !?[]u8 {
    if (std.mem.indexOfScalar(u8, id, '/') != null or std.mem.endsWith(u8, id, ".jsonl")) {
        std.Io.Dir.cwd().access(io, id, .{}) catch return null;
        return try alloc.dupe(u8, id);
    }
    if (std.mem.eql(u8, id, latest)) {
        const sessions = try list(alloc, io, root_dir, cwd);
        defer freeList(alloc, sessions);
        if (sessions.len == 0) return null;
        return try alloc.dupe(u8, sessions[0].path);
    }
    const dir_path = try sessionsPath(alloc, root_dir, cwd);
    defer alloc.free(dir_path);
    var dir = std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer dir.close(io);
    var iterator = dir.iterate();
    while (try iterator.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".jsonl")) continue;
        const stem = entry.name[0 .. entry.name.len - ".jsonl".len];
        const underscore = std.mem.lastIndexOfScalar(u8, stem, '_') orelse continue;
        if (!std.mem.eql(u8, stem[underscore + 1 ..], id)) continue;
        return try std.fs.path.join(alloc, &.{ dir_path, entry.name });
    }
    return null;
}

/// What an id or id prefix names among the workspace's sessions.
pub const Match = union(enum) {
    none,
    one: Summary,
    /// More than one: the ids, newest first, to say which were meant.
    many: []Summary,

    pub fn deinit(self: Match, alloc: Allocator) void {
        switch (self) {
            .none => {},
            .one => |s| s.deinit(alloc),
            .many => |list_| freeList(alloc, list_),
        }
    }
};

/// The session `query` names: a whole id, or a prefix of exactly one.
pub fn match(alloc: Allocator, io: std.Io, root_dir: []const u8, cwd: []const u8, query: []const u8) !Match {
    const sessions = try list(alloc, io, root_dir, cwd);
    var kept: usize = 0;
    for (sessions) |s| {
        if (std.mem.startsWith(u8, s.id, query)) {
            sessions[kept] = s;
            kept += 1;
        } else s.deinit(alloc);
    }
    if (kept == 0) {
        alloc.free(sessions);
        return .none;
    }
    // A whole id is never ambiguous, even when it prefixes no other.
    if (kept == 1) {
        const one = sessions[0];
        alloc.free(sessions);
        return .{ .one = one };
    }
    const shrunk = alloc.realloc(sessions, kept) catch sessions[0..kept];
    return .{ .many = shrunk };
}

/// Deletes the session file at `path` and the token-cache states its turn
/// boundaries name under `<root>/cache/prefix/` (`/save` exports stay).
/// Returns the states removed. A file that cannot be read is still deleted.
pub fn delete(alloc: Allocator, io: std.Io, root_dir: []const u8, path: []const u8) !usize {
    var removed: usize = 0;
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    if (session.load(arena.allocator(), io, .cwd(), path, null)) |loaded| {
        const cache_dir = try paths.prefixCachePath(alloc, root_dir);
        defer alloc.free(cache_dir);
        for (loaded.records) |record| switch (record.entry) {
            .assistant => |step| if (step.boundary) |b| {
                removed += cache.removeDigest(io, cache_dir, b.digest);
            },
            else => {},
        };
    } else |_| {}
    try std.Io.Dir.cwd().deleteFile(io, path);
    return removed;
}

/// The stored conversation's last saved turn end, for the completer to
/// restore from the token cache instead of replaying.
pub fn lastBoundary(loaded: session.Loaded) ?cache.Boundary {
    var i = loaded.records.len;
    while (i > 0) {
        i -= 1;
        switch (loaded.records[i].entry) {
            .assistant => |step| if (step.boundary) |b| return .{ .bytes = b.bytes, .digest = b.digest },
            else => {},
        }
    }
    return null;
}

/// The `/list` rows: one per session, newest first, `current` (an id)
/// marked. Strings in `alloc` (an arena, in the agent).
pub fn listRows(alloc: Allocator, sessions: []const Summary, current: []const u8, ascii: bool) ![]const []const u8 {
    var rows: std.ArrayList([]const u8) = .empty;
    if (sessions.len == 0) {
        try rows.append(alloc, "  no saved sessions for this workspace");
        return rows.toOwnedSlice(alloc);
    }
    try rows.append(alloc, "  saved sessions, newest first (/resume continues one, /delete removes one)");
    for (sessions) |s| {
        const short = if (s.id.len > 8) s.id[0..8] else s.id;
        const marker: []const u8 = if (std.mem.eql(u8, s.id, current)) (if (ascii) "*" else "●") else " ";
        try rows.append(alloc, try alloc.print("  {s} {s}  {s}  {s: <6} ctx {d: <5}  {s}", .{ marker, short, s.time, s.effort, s.ctx_size, s.first_prompt }));
    }
    return rows.toOwnedSlice(alloc);
}

fn sessionsPath(alloc: Allocator, root_dir: []const u8, cwd: []const u8) ![]u8 {
    const slug = try session.cwdSlug(alloc, cwd);
    defer alloc.free(slug);
    return std.fs.path.join(alloc, &.{ root_dir, paths.agent_dir, paths.sessions_dir, slug });
}

/// The conversation the loop renders, in order. Non-conversation entries
/// (effort, context, compaction, notice) are skipped: they are session facts,
/// not messages. Slices borrow `loaded`.
pub fn messages(alloc: Allocator, loaded: session.Loaded) ![]Profile.Message {
    var out: std.ArrayList(Profile.Message) = .empty;
    errdefer out.deinit(alloc);
    for (loaded.records) |record| {
        switch (record.entry) {
            .user => |u| try out.append(alloc, .{ .role = .user, .content = u.text }),
            .assistant => |a| try out.append(alloc, .{
                .role = .assistant,
                .content = a.answer,
                .reasoning_content = a.thinking,
                .tool_calls = a.tool_calls,
            }),
            .tool_result => |r| try out.append(alloc, .{
                .role = .tool,
                .content = r.text,
                .tool_call_id = std.fmt.parseInt(u32, r.call, 10) catch null,
            }),
            else => {},
        }
    }
    return out.toOwnedSlice(alloc);
}

/// Applies a stored conversation to a transcript and returns the rows it
/// closed, for the caller to insert above the live region. The turn boundary
/// is recovered from the user entries: the live path emits one `turn_end` at
/// the true end of a turn, which the file records only as the next `user`.
pub fn replay(alloc: Allocator, tr: *transcript.Transcript, loaded: session.Loaded, options: transcript.Render) ![]const screen.Row {
    var out: std.ArrayList(screen.Row) = .empty;
    errdefer out.deinit(alloc);
    var open_turn = false;
    for (loaded.records) |record| {
        switch (record.entry) {
            .user => |u| {
                if (open_turn) try tr.apply(.{ .turn_end = .{ .stop = .eos } });
                try tr.apply(.{ .user = u.text });
                for (u.images, 1..) |image, n| {
                    const label = try alloc.print("image #{d}: {s} ({d}×{d} → {d}×{d} tokens)", .{ n, image.path, image.width, image.height, image.width_tokens, image.height_tokens });
                    defer alloc.free(label);
                    try tr.apply(.{ .attachment = .{ .label = label } });
                }
                open_turn = true;
            },
            .assistant => |a| {
                if (a.thinking.len > 0) {
                    try tr.apply(.{ .thinking_delta = a.thinking });
                    try tr.apply(.{ .thinking_end = a.stats.thinking_seconds });
                }
                if (a.answer.len > 0) try tr.apply(.{ .answer_delta = a.answer });
                for (a.tool_calls) |call| {
                    var described = try tools.describe(alloc, call.name, call.arguments);
                    defer described.deinit(alloc);
                    try tr.apply(.{ .tool_call = .{ .id = call.id, .name = call.name, .summary = described.summary, .detail = described.detail } });
                }
                tr.closeOpen();
                open_turn = true;
            },
            .tool_result => |r| try tr.apply(.{ .tool_result = .{
                .id = std.fmt.parseInt(u32, r.call, 10) catch 0,
                .text = r.text,
                .truncated = r.truncated,
                .is_error = r.is_error,
                .summary = r.summary,
            } }),
            else => {},
        }
        try out.appendSlice(alloc, try tr.takeClosed(alloc, options));
    }
    if (open_turn) try tr.apply(.{ .turn_end = .{ .stop = .eos } });
    try out.appendSlice(alloc, try tr.takeClosed(alloc, options));
    return out.toOwnedSlice(alloc);
}

// ----- tests -----

const testing = std.testing;

fn sampleSession(alloc: Allocator) !session.Loaded {
    const text = "{\"type\":\"session\",\"version\":1,\"id\":\"abc\",\"time\":\"2026-09-14T10:00:00Z\",\"cwd\":\"/w\",\"model\":{\"path\":\"/m.gguf\"},\"effort\":\"low\",\"ctx_size\":8192}\n" ++
        "{\"type\":\"user\",\"id\":1,\"parent\":null,\"time\":\"t1\",\"text\":\"add a greeting\"}\n" ++
        "{\"type\":\"assistant\",\"id\":2,\"parent\":1,\"time\":\"t2\",\"thinking\":\"read it first\",\"answer\":\"Let me look.\\n\"," ++
        "\"tool_calls\":[{\"id\":4,\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"main.zig\\\"}\"}],\"stop\":\"eos\",\"stats\":{\"generated\":5}}\n" ++
        "{\"type\":\"tool_result\",\"id\":3,\"parent\":2,\"time\":\"t3\",\"call\":\"4\",\"text\":\"pub fn main() {}\",\"truncated\":false,\"is_error\":false,\"summary\":\"lines 1 to 1 of 1\"}\n" ++
        "{\"type\":\"assistant\",\"id\":4,\"parent\":3,\"time\":\"t4\",\"answer\":\"Added it.\\n\",\"stop\":\"eos\",\"stats\":{\"generated\":3}}\n";
    return session.parseText(alloc, text, null);
}

test "the conversation is rebuilt from the entries, tool exchange included" {
    const alloc = testing.allocator;
    const loaded = try sampleSession(alloc);
    defer loaded.deinit();
    const built = try messages(alloc, loaded);
    defer alloc.free(built);
    try testing.expectEqual(@as(usize, 4), built.len);
    try testing.expectEqual(Profile.Role.user, built[0].role);
    try testing.expectEqual(Profile.Role.assistant, built[1].role);
    try testing.expectEqual(@as(usize, 1), built[1].tool_calls.len);
    try testing.expectEqual(@as(u32, 4), built[1].tool_calls[0].id);
    try testing.expectEqualStrings("read_file", built[1].tool_calls[0].name);
    try testing.expectEqual(Profile.Role.tool, built[2].role);
    try testing.expectEqual(@as(u32, 4), built[2].tool_call_id.?);
    try testing.expectEqualStrings("Added it.\n", built[3].content);
    // Non-conversation entries never become messages.
    try testing.expectEqual(@as(usize, 0), built[3].tool_calls.len);
}

test "the replay renders the saved conversation, tool call and result included" {
    const alloc = testing.allocator;
    const loaded = try sampleSession(alloc);
    defer loaded.deinit();
    var tr: transcript.Transcript = .{ .alloc = alloc };
    defer tr.deinit();
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const rows = try replay(arena.allocator(), &tr, loaded, .{ .width = 60, .th = .{ .kind = .plain } });
    var text: std.ArrayList(u8) = .empty;
    for (rows) |row| {
        try text.appendSlice(arena.allocator(), row.text);
        try text.append(arena.allocator(), '\n');
    }
    const s = text.items;
    try testing.expect(std.mem.indexOf(u8, s, "add a greeting") != null);
    try testing.expect(std.mem.indexOf(u8, s, "Let me look.") != null);
    try testing.expect(std.mem.indexOf(u8, s, "Read(main.zig)") != null);
    // The result's text stays in the file; the transcript shows its summary.
    try testing.expect(std.mem.indexOf(u8, s, "pub fn main() {}") == null);
    try testing.expect(std.mem.indexOf(u8, s, "lines 1 to 1 of 1") != null);
    try testing.expect(std.mem.indexOf(u8, s, "Added it.") != null);
    // The tool result follows the call it answers.
    const answer_at = std.mem.indexOf(u8, s, "Let me look.") orelse return error.TestUnexpectedResult;
    const call_at = std.mem.indexOf(u8, s, "Read(main.zig)") orelse return error.TestUnexpectedResult;
    const result_at = std.mem.indexOf(u8, s, "lines 1 to 1 of 1") orelse return error.TestUnexpectedResult;
    try testing.expect(answer_at < call_at);
    try testing.expect(call_at < result_at);
}

test "listing finds sessions for the working directory and orders them newest first" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const alloc = testing.allocator;
    const io = testing.io;
    const root = try tmp.dir.realPathFileAlloc(io, ".", alloc);
    defer alloc.free(root);
    const cwd = "/work/proj";
    const slug = try session.cwdSlug(alloc, cwd);
    defer alloc.free(slug);
    const dir = try std.fs.path.join(alloc, &.{ root, "agent", "sessions", slug });
    defer alloc.free(dir);
    try std.Io.Dir.cwd().createDirPath(io, dir);
    const first = "{\"type\":\"session\",\"version\":1,\"id\":\"aaaa\",\"time\":\"2026-09-14T09:00:00Z\",\"cwd\":\"/work/proj\",\"model\":{\"path\":\"/m\"},\"effort\":\"low\",\"ctx_size\":8192}\n";
    const second = "{\"type\":\"session\",\"version\":1,\"id\":\"bbbb\",\"time\":\"2026-09-14T11:30:00Z\",\"cwd\":\"/work/proj\",\"model\":{\"path\":\"/m\"},\"effort\":\"medium\",\"ctx_size\":16384}\n";
    const first_rel = try std.fs.path.join(alloc, &.{ "agent", "sessions", slug, "20260914T090000Z_aaaa.jsonl" });
    defer alloc.free(first_rel);
    try tmp.dir.writeFile(io, .{ .sub_path = first_rel, .data = first });
    const second_rel = try std.fs.path.join(alloc, &.{ "agent", "sessions", slug, "20260914T113000Z_bbbb.jsonl" });
    defer alloc.free(second_rel);
    try tmp.dir.writeFile(io, .{ .sub_path = second_rel, .data = second });

    const items = try list(alloc, io, root, cwd);
    defer freeList(alloc, items);
    try testing.expectEqual(@as(usize, 2), items.len);
    try testing.expectEqualStrings("bbbb", items[0].id);
    try testing.expectEqualStrings("2026-09-14T11:30:00Z", items[0].time);
    try testing.expectEqualStrings("medium", items[0].effort);
    try testing.expectEqual(@as(usize, 16384), items[0].ctx_size);
    try testing.expectEqualStrings("aaaa", items[1].id);
    try testing.expect(std.mem.endsWith(u8, items[0].path, "20260914T113000Z_bbbb.jsonl"));

    const found = try find(alloc, io, root, cwd, "aaaa");
    defer if (found) |path| alloc.free(path);
    try testing.expect(found != null);
    try testing.expect(std.mem.endsWith(u8, found.?, "20260914T090000Z_aaaa.jsonl"));
    try testing.expect((try find(alloc, io, root, cwd, "cccc")) == null);
    // An explicit path is used as given, so print mode can script one file.
    const abs_second = try std.fs.path.join(alloc, &.{ root, second_rel });
    defer alloc.free(abs_second);
    const by_path = try find(alloc, io, root, cwd, abs_second);
    defer if (by_path) |path| alloc.free(path);
    try testing.expect(by_path != null);
    try testing.expectEqualStrings(abs_second, by_path.?);
    try testing.expect((try find(alloc, io, root, cwd, "/no/such/file.jsonl")) == null);
    // A workspace with no sessions is an empty list, not an error.
    const none = try list(alloc, io, root, "/work/other");
    defer freeList(alloc, none);
    try testing.expectEqual(@as(usize, 0), none.len);
}

test "an id prefix names one session or says which it could mean, and delete takes its cached states" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const alloc = testing.allocator;
    const io = testing.io;
    const root = try tmp.dir.realPathFileAlloc(io, ".", alloc);
    defer alloc.free(root);
    const cwd = "/work/proj";
    const slug = try session.cwdSlug(alloc, cwd);
    defer alloc.free(slug);
    const header = "{{\"type\":\"session\",\"version\":1,\"id\":\"{s}\",\"time\":\"{s}\",\"cwd\":\"/work/proj\",\"model\":{{\"path\":\"/m\"}},\"effort\":\"low\",\"ctx_size\":8192}}\n";
    const ids = [_][]const u8{ "ab12", "ab34", "cd56" };
    const times = [_][]const u8{ "2026-10-04T09:00:00Z", "2026-10-04T10:00:00Z", "2026-10-04T11:00:00Z" };
    for (ids, times) |id, time| {
        const rel = try std.fmt.allocPrint(alloc, "agent/sessions/{s}/x_{s}.jsonl", .{ slug, id });
        defer alloc.free(rel);
        var body: std.ArrayList(u8) = .empty;
        defer body.deinit(alloc);
        try body.print(alloc, header, .{ id, time });
        if (std.mem.eql(u8, id, "cd56")) try body.appendSlice(alloc, "{\"type\":\"assistant\",\"id\":1,\"parent\":null,\"time\":\"t\",\"answer\":\"a\",\"stop\":\"eos\",\"boundary\":{\"bytes\":9,\"digest\":42}}\n");
        try tmp.dir.createDirPath(io, std.fs.path.dirname(rel).?);
        try tmp.dir.writeFile(io, .{ .sub_path = rel, .data = body.items });
    }
    // Two cached states for the boundary (another model's too), one unrelated.
    try tmp.dir.createDirPath(io, "cache/prefix");
    for ([_][]const u8{ "0000000000000001-000000000000002a-0000000000000007.snap", "0000000000000002-000000000000002a-0000000000000009.snap", "0000000000000001-000000000000002b-0000000000000007.snap" }) |name| {
        const rel = try std.fmt.allocPrint(alloc, "cache/prefix/{s}", .{name});
        defer alloc.free(rel);
        try tmp.dir.writeFile(io, .{ .sub_path = rel, .data = "x" });
    }

    const ambiguous = try match(alloc, io, root, cwd, "ab");
    defer ambiguous.deinit(alloc);
    try testing.expectEqual(@as(usize, 2), ambiguous.many.len);
    try testing.expectEqualStrings("ab34", ambiguous.many[0].id);
    const none = try match(alloc, io, root, cwd, "zz");
    try testing.expect(none == .none);
    const one = try match(alloc, io, root, cwd, "cd");
    defer one.deinit(alloc);
    try testing.expectEqualStrings("cd56", one.one.id);

    try testing.expectEqual(@as(usize, 2), try delete(alloc, io, root, one.one.path));
    const left = try list(alloc, io, root, cwd);
    defer freeList(alloc, left);
    try testing.expectEqual(@as(usize, 2), left.len);
    try tmp.dir.access(io, "cache/prefix/0000000000000001-000000000000002b-0000000000000007.snap", .{});

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const rows = try listRows(arena.allocator(), left, "ab12", false);
    try testing.expectEqual(@as(usize, 3), rows.len);
    try testing.expect(std.mem.indexOf(u8, rows[2], "● ab12") != null);
    try testing.expect(std.mem.indexOf(u8, rows[1], "  ab34") != null);
}

test {
    _ = inference;
}
