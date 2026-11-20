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
const loop = @import("loop.zig");
const handoff = @import("handoff.zig");
const kit = @import("../ui.zig");

const Allocator = std.mem.Allocator;
const Profile = inference.profiles;
const transcript = tui.transcript;
const screen = tui.screen;

/// The id `find` resolves to the newest session of the workspace: `--resume
/// latest`, and `--resume` with no id in print mode. A real id is 32 hex
/// digits, so the word cannot collide.
pub const latest = "latest";

/// What a bare `--resume` asks of the terminal surface: the session picker
/// at startup. Never passed to `find`; print mode turns it into `latest`.
pub const pick = "";

/// One saved conversation, as a picker sees it: the header's facts plus the
/// path to load. Owned strings.
pub const Summary = struct {
    id: []u8,
    time: []u8,
    cwd: []u8,
    effort: []u8,
    ctx_size: usize,
    /// The file's size on disk.
    bytes: u64,
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

    /// The text form: a header with the workspace, one row per session
    /// (its age against `now`, seconds since the epoch), and how to resume.
    pub fn render(self: Listing, arena: Allocator, out: *std.Io.Writer, json: bool, ui: kit.Ui, now: i64) !void {
        if (json) {
            try std.json.Stringify.value(self, .{ .whitespace = .indent_2 }, out);
            return out.writeByte('\n');
        }
        const v = kit.values;
        try kit.blocks.header(out, ui, "sessions", try std.fmt.allocPrint(arena, "{s} · {f}", .{ try kit.text.clean(arena, self.cwd), v.count(self.sessions.len, "session") }));
        if (self.sessions.len == 0) {
            try kit.blocks.result(out, ui, .not_checked, "no saved sessions for this workspace", "");
            return kit.blocks.hint(out, ui, "", "nuclis agent", " starts one");
        }
        const columns = [_]kit.table.Column{
            .{ .title = "ID", .truncate = false },
            .{ .title = "STARTED" },
            .{ .title = "EFFORT", .drop = 2 },
            .{ .title = "CTX", .alignment = .right, .drop = 1 },
            .{ .title = "PROMPT", .min = 12 },
        };
        const rows = try arena.alloc([]const kit.table.Cell, self.sessions.len);
        for (self.sessions, rows) |s, *row| row.* = try arena.dupe(kit.table.Cell, &.{
            .{ .text = try kit.text.clean(arena, s.id[0..@min(8, s.id.len)]), .role = .id },
            .{ .text = if (v.parseUtc(s.time)) |t| try std.fmt.allocPrint(arena, "{f}", .{v.ago(now - t)}) else try kit.text.clean(arena, s.time) },
            .{ .text = try kit.text.clean(arena, s.effort) },
            .{ .text = try std.fmt.allocPrint(arena, "{f}", .{v.grouped(s.ctx_size)}) },
            .{ .text = try kit.text.clean(arena, s.first_prompt), .role = .muted },
        });
        try kit.table.render(out, ui, arena, 0, &columns, rows);
        try out.writeByte('\n');
        try kit.blocks.hint(out, ui, "", "nuclis agent --resume <id>", " continues one; --resume alone picks");
    }
};

/// `nuclis agent ls`: the sessions recorded for `cwd`.
pub fn ls(alloc: Allocator, io: std.Io, root_dir: []const u8, cwd: []const u8, json: bool, out: *std.Io.Writer, ui: kit.Ui) !void {
    const sessions = try list(alloc, io, root_dir, cwd);
    defer freeList(alloc, sessions);
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const listing: Listing = .{ .cwd = cwd, .sessions = sessions };
    try listing.render(arena_state.allocator(), out, json, ui, std.Io.Timestamp.now(io, .real).toSeconds());
}

pub fn freeList(alloc: Allocator, items: []Summary) void {
    for (items) |item| item.deinit(alloc);
    alloc.free(items);
}

/// How much of a session file `list` reads: the header and the first user
/// entry. A first prompt cut off by it shows no preview.
const summary_prefix_bytes = 64 * 1024;

/// The sessions recorded for `cwd`, newest first, whatever their size: each
/// file is read only up to `summary_prefix_bytes`. A root with no session
/// directory yet is an empty list, not an error; a file whose first line is
/// not a header is skipped.
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
    const buffer = try alloc.alloc(u8, summary_prefix_bytes);
    defer alloc.free(buffer);
    var iterator = dir.iterate();
    while (try iterator.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".jsonl")) continue;
        const head = readHead(io, dir, entry.name, buffer) catch continue;
        const header = session.parseHeader(alloc, head.bytes) catch continue;
        defer header.deinit(alloc);
        const path = try std.fs.path.join(alloc, &.{ dir_path, entry.name });
        errdefer alloc.free(path);
        const first_prompt = try firstPrompt(alloc, head.bytes);
        errdefer alloc.free(first_prompt);
        const id = try alloc.dupe(u8, header.id);
        errdefer alloc.free(id);
        const time = try alloc.dupe(u8, header.time);
        errdefer alloc.free(time);
        const directory = try alloc.dupe(u8, header.cwd);
        errdefer alloc.free(directory);
        const effort = try alloc.dupe(u8, header.effort);
        errdefer alloc.free(effort);
        try out.append(alloc, .{
            .id = id,
            .time = time,
            .cwd = directory,
            .effort = effort,
            .ctx_size = header.ctx_size,
            .bytes = head.size,
            .path = path,
            .first_prompt = first_prompt,
        });
    }
    std.mem.sort(Summary, out.items, {}, newerFirst);
    return out.toOwnedSlice(alloc);
}

/// The first bytes of `name` (at most `buffer.len`, borrowed from it) and
/// the file's whole size.
fn readHead(io: std.Io, dir: std.Io.Dir, name: []const u8, buffer: []u8) !struct { bytes: []const u8, size: u64 } {
    const file = try dir.openFile(io, name, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    const n = try file.readPositionalAll(io, buffer, 0);
    return .{ .bytes = buffer[0..n], .size = stat.size };
}

/// The first line of the first user entry, cut to the preview length at a
/// code point; empty when the first entry past the header and any `system`
/// entries is not a whole user entry (`bytes` may be a prefix that ends
/// mid-line).
fn firstPrompt(alloc: Allocator, bytes: []const u8) ![]u8 {
    var start = (std.mem.indexOfScalar(u8, bytes, '\n') orelse return alloc.dupe(u8, "")) + 1;
    const Line = struct { type: []const u8, text: []const u8 = "" };
    const parsed = while (true) {
        const end = std.mem.indexOfScalarPos(u8, bytes, start, '\n') orelse bytes.len;
        const line = std.json.parseFromSlice(Line, alloc, bytes[start..end], .{ .ignore_unknown_fields = true }) catch return alloc.dupe(u8, "");
        if (!std.mem.eql(u8, line.value.type, "system") or end == bytes.len) break line;
        line.deinit();
        start = end + 1;
    };
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
    // A copy, not a shrink in place: the caller frees by the returned length.
    const shrunk = alloc.dupe(Summary, sessions[0..kept]) catch |err| {
        for (sessions[0..kept]) |s| s.deinit(alloc);
        alloc.free(sessions);
        return err;
    };
    alloc.free(sessions);
    return .{ .many = shrunk };
}

/// Deletes the session file at `path` and the token-cache states its turn
/// boundaries name under `<root>/cache/prefix/` (exports stay).
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

/// The directory holding `cwd`'s session files under `root_dir`. Caller owns it.
pub fn sessionsPath(alloc: Allocator, root_dir: []const u8, cwd: []const u8) ![]u8 {
    const slug = try session.cwdSlug(alloc, cwd);
    defer alloc.free(slug);
    return std.fs.path.join(alloc, &.{ root_dir, paths.agent_dir, paths.sessions_dir, slug });
}

/// The conversation the model saw when the session was written: its
/// messages in order, and for each the index in `loaded.records` it came
/// from, so the caller can attach a user entry's media.
pub const Conversation = struct {
    messages: []Profile.Message,
    records: []usize,
    /// The elision stubs some messages' content or call arguments point
    /// at, and the call arrays rebuilt to hold shortened arguments; owned.
    stubs: [][]u8,
    calls: [][]Profile.ToolCall = &.{},
    /// The latest recorded system block (`loop.Agent.adoptSystem`), or null
    /// for a file from before it was recorded; borrows `loaded`.
    system: ?[]const u8 = null,

    pub fn deinit(self: Conversation, alloc: Allocator) void {
        for (self.stubs) |stub| alloc.free(stub);
        alloc.free(self.stubs);
        for (self.calls) |list_| alloc.free(list_);
        alloc.free(self.calls);
        alloc.free(self.records);
        alloc.free(self.messages);
    }
};

/// Rebuilds what the live loop held: a cancelled step never reached the
/// history, and each compaction (`loop.Compaction`) is applied where it was
/// recorded. Non-conversation entries (effort, context, notice) are session
/// facts, not messages. Slices other than the stubs borrow `loaded`.
pub fn conversation(alloc: Allocator, loaded: session.Loaded) !Conversation {
    var out: std.ArrayList(Profile.Message) = .empty;
    defer out.deinit(alloc);
    var at: std.ArrayList(usize) = .empty;
    defer at.deinit(alloc);
    var stubs: std.ArrayList([]u8) = .empty;
    defer {
        for (stubs.items) |stub| alloc.free(stub);
        stubs.deinit(alloc);
    }
    var calls: std.ArrayList([]Profile.ToolCall) = .empty;
    defer {
        for (calls.items) |list_| alloc.free(list_);
        calls.deinit(alloc);
    }
    // Where the turn in progress begins in `out`, as the loop's `turn_start`.
    var turn_start: usize = 0;
    var system: ?[]const u8 = null;
    for (loaded.records, 0..) |record, index| {
        const message: Profile.Message = switch (record.entry) {
            .user => |u| blk: {
                turn_start = out.items.len;
                break :blk .{ .role = .user, .content = u.text };
            },
            .assistant => |a| if (std.mem.eql(u8, a.stop, "cancelled")) continue else .{
                .role = .assistant,
                .content = a.answer,
                .reasoning_content = a.thinking,
                .tool_calls = a.tool_calls,
            },
            .tool_result => |r| .{
                .role = .tool,
                .content = r.text,
                .tool_call_id = std.fmt.parseInt(u32, r.call, 10) catch null,
            },
            .compaction => |c| {
                const kept = @min(c.first_kept, out.items.len);
                if (std.mem.eql(u8, c.reason, "context_full")) {
                    out.replaceRangeAssumeCapacity(0, kept, &.{});
                    at.replaceRangeAssumeCapacity(0, kept, &.{});
                    turn_start -|= kept;
                } else if (std.mem.eql(u8, c.reason, "results_elided")) {
                    for (out.items[@min(turn_start, kept)..kept]) |*m| {
                        if (m.role != .tool or std.mem.startsWith(u8, m.content, loop.elided_prefix)) continue;
                        try stubs.ensureUnusedCapacity(alloc, 1);
                        const stub = try loop.elisionStub(alloc, loop.callName(out.items, m.tool_call_id), m.content);
                        stubs.appendAssumeCapacity(stub);
                        m.content = stub;
                    }
                } else if (std.mem.eql(u8, c.reason, "turn_aborted")) {
                    out.shrinkRetainingCapacity(kept);
                    at.shrinkRetainingCapacity(kept);
                    turn_start = @min(turn_start, kept);
                } else if (std.mem.eql(u8, c.reason, loop.reasoning_dropped)) {
                    for (out.items[0..kept]) |*m| if (m.role == .assistant) {
                        m.reasoning_content = "";
                    };
                } else if (std.mem.eql(u8, c.reason, loop.arguments_stubbed)) {
                    for (out.items[0..kept]) |*m| try stubCalls(alloc, m, &stubs, &calls);
                }
                continue;
            },
            .system => |sys| {
                system = sys.text;
                continue;
            },
            .handoff => |h| {
                out.clearRetainingCapacity();
                at.clearRetainingCapacity();
                turn_start = 0;
                try out.append(alloc, .{ .role = .user, .content = h.text });
                try at.append(alloc, index);
                continue;
            },
            else => continue,
        };
        try out.ensureUnusedCapacity(alloc, 1);
        try at.ensureUnusedCapacity(alloc, 1);
        out.appendAssumeCapacity(message);
        at.appendAssumeCapacity(index);
    }
    const messages = try out.toOwnedSlice(alloc);
    errdefer alloc.free(messages);
    const records = try at.toOwnedSlice(alloc);
    errdefer alloc.free(records);
    const owned_stubs = try stubs.toOwnedSlice(alloc);
    errdefer {
        for (owned_stubs) |stub| alloc.free(stub);
        alloc.free(owned_stubs);
    }
    return .{ .messages = messages, .records = records, .stubs = owned_stubs, .calls = try calls.toOwnedSlice(alloc), .system = system };
}

/// Shortens `m`'s write and edit calls as the loop did
/// (`handoff.argumentStub`), into a call array of its own.
fn stubCalls(alloc: Allocator, m: *Profile.Message, stubs: *std.ArrayList([]u8), calls: *std.ArrayList([]Profile.ToolCall)) !void {
    if (m.role != .assistant or m.tool_calls.len == 0) return;
    var copy: []Profile.ToolCall = &.{};
    for (m.tool_calls, 0..) |call, i| {
        try stubs.ensureUnusedCapacity(alloc, 1);
        const stub = try handoff.argumentStub(alloc, call.name, call.arguments) orelse continue;
        stubs.appendAssumeCapacity(stub);
        if (copy.len == 0) {
            try calls.ensureUnusedCapacity(alloc, 1);
            copy = try alloc.dupe(Profile.ToolCall, m.tool_calls);
            calls.appendAssumeCapacity(copy);
        }
        copy[i].arguments = stub;
    }
    if (copy.len > 0) m.tool_calls = copy;
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
                for (u.audio, 1..) |clip, n| {
                    const label = try alloc.print("audio #{d}: {s} ({d:.1} s → {d} tokens)", .{ n, clip.path, clip.seconds(), clip.tokens });
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
                .cancelled = r.cancelled,
                .summary = r.summary,
            } }),
            // Display only: the rows above stay, the model resumes from here.
            .handoff => try tr.apply(.{ .notice = "  — handed off here: the model continues from a summary of the above" }),
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
    const conv = try conversation(alloc, loaded);
    defer conv.deinit(alloc);
    const built = conv.messages;
    try testing.expectEqual(@as(usize, 4), built.len);
    try testing.expectEqualSlices(usize, &.{ 0, 1, 2, 3 }, conv.records);
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

test "a resume sees what the model saw: no cancelled step, every compaction applied" {
    const alloc = testing.allocator;
    const head = "{\"type\":\"session\",\"version\":1,\"id\":\"abc\",\"time\":\"t\",\"cwd\":\"/w\",\"model\":{\"path\":\"/m.gguf\"},\"effort\":\"low\",\"ctx_size\":8192}\n";
    const text = head ++
        // Turn one, then a cancelled step of turn two: the history never held it.
        "{\"type\":\"user\",\"id\":1,\"parent\":null,\"time\":\"t\",\"text\":\"one\"}\n" ++
        "{\"type\":\"assistant\",\"id\":2,\"parent\":1,\"time\":\"t\",\"answer\":\"first\",\"stop\":\"eos\"}\n" ++
        "{\"type\":\"user\",\"id\":3,\"parent\":2,\"time\":\"t\",\"text\":\"two\"}\n" ++
        "{\"type\":\"assistant\",\"id\":4,\"parent\":3,\"time\":\"t\",\"answer\":\"half\",\"stop\":\"cancelled\"}\n" ++
        // Turn three reads twice; turn one is dropped, then the first read elided.
        "{\"type\":\"user\",\"id\":5,\"parent\":4,\"time\":\"t\",\"text\":\"three\"}\n" ++
        "{\"type\":\"assistant\",\"id\":6,\"parent\":5,\"time\":\"t\",\"answer\":\"\",\"tool_calls\":[{\"id\":7,\"name\":\"read_file\",\"arguments\":\"{}\"},{\"id\":8,\"name\":\"grep\",\"arguments\":\"{}\"}],\"stop\":\"eos\"}\n" ++
        "{\"type\":\"tool_result\",\"id\":7,\"parent\":6,\"time\":\"t\",\"call\":\"7\",\"text\":\"a\\nb\\nc\",\"truncated\":false,\"is_error\":false}\n" ++
        "{\"type\":\"tool_result\",\"id\":8,\"parent\":6,\"time\":\"t\",\"call\":\"8\",\"text\":\"hit\",\"truncated\":false,\"is_error\":false}\n" ++
        // History then: one, first, two, three, calls, r7, r8.
        "{\"type\":\"compaction\",\"id\":9,\"parent\":8,\"time\":\"t\",\"first_kept\":2,\"reason\":\"context_full\"}\n" ++
        // Now: two, three, calls, r7, r8; turn three starts at 1.
        "{\"type\":\"compaction\",\"id\":10,\"parent\":9,\"time\":\"t\",\"first_kept\":4,\"reason\":\"results_elided\"}\n" ++
        "{\"type\":\"assistant\",\"id\":11,\"parent\":10,\"time\":\"t\",\"answer\":\"done\",\"stop\":\"eos\"}\n" ++
        // A failed fourth turn, cut from the history.
        "{\"type\":\"user\",\"id\":12,\"parent\":11,\"time\":\"t\",\"text\":\"four\"}\n" ++
        "{\"type\":\"compaction\",\"id\":13,\"parent\":12,\"time\":\"t\",\"first_kept\":6,\"reason\":\"turn_aborted\"}\n";
    const loaded = try session.parseText(alloc, text, null);
    defer loaded.deinit();
    const conv = try conversation(alloc, loaded);
    defer conv.deinit(alloc);
    const m = conv.messages;
    try testing.expectEqual(@as(usize, 6), m.len);
    try testing.expectEqualStrings("two", m[0].content);
    try testing.expectEqualStrings("three", m[1].content);
    try testing.expectEqualStrings("[result elided to fit the context: read_file, 3 lines]", m[3].content);
    try testing.expectEqualStrings("hit", m[4].content);
    try testing.expectEqualStrings("done", m[5].content);
    // The fourth record of the file is the cancelled step: no message is from it.
    try testing.expectEqualSlices(usize, &.{ 2, 4, 5, 6, 7, 10 }, conv.records);
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
    // The text form: newest first, each start as an age.
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    var text: std.Io.Writer.Allocating = .init(alloc);
    defer text.deinit();
    const listing: Listing = .{ .cwd = cwd, .sessions = items };
    try listing.render(arena_state.allocator(), &text.writer, false, .plain, kit.values.parseUtc("2026-09-14T11:32:00Z").?);
    try testing.expectEqualStrings(
        \\◆ sessions · /work/proj · 2 sessions
        \\
        \\ID    STARTED  EFFORT     CTX
        \\bbbb  2m ago   medium  16,384
        \\aaaa  2h ago   low      8,192
        \\
        \\→ nuclis agent --resume <id> continues one; --resume alone picks
        \\
    , text.written());
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

test "the preview is the first user entry, past the system block" {
    const alloc = std.testing.allocator;
    const preview = try firstPrompt(alloc, "{\"type\":\"session\"}\n{\"type\":\"system\",\"text\":\"You are nuclis.\"}\n{\"type\":\"user\",\"text\":\"build a todo app\\nnow\"}\n");
    defer alloc.free(preview);
    try std.testing.expectEqualStrings("build a todo app", preview);
    const none = try firstPrompt(alloc, "{\"type\":\"session\"}\n{\"type\":\"system\",\"text\":\"x\"}\n{\"type\":\"assistant\"}\n");
    defer alloc.free(none);
    try std.testing.expectEqualStrings("", none);
}

test "listing reads a session of any size, skips a damaged header, and keeps newest first" {
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
    const user = "{\"type\":\"user\",\"id\":1,\"parent\":null,\"time\":\"t\",\"text\":\"fix the parser\\nplease\"}\n";
    const answer = "{\"type\":\"assistant\",\"id\":2,\"parent\":1,\"time\":\"t\",\"answer\":\"" ++ @as([1000]u8, @splat('a')) ++ "\",\"stop\":\"eos\"}\n";
    const dir = try std.fmt.allocPrint(alloc, "agent/sessions/{s}", .{slug});
    defer alloc.free(dir);
    try tmp.dir.createDirPath(io, dir);

    // Oldest: past 200 KiB, well beyond the summary prefix.
    var big: std.ArrayList(u8) = .empty;
    defer big.deinit(alloc);
    try big.print(alloc, header, .{ "1111", "2026-11-01T09:00:00Z" });
    try big.appendSlice(alloc, user);
    while (big.items.len < 200 * 1024) try big.appendSlice(alloc, answer);
    // Newest: a first prompt longer than the summary bound has no preview.
    var long: std.ArrayList(u8) = .empty;
    defer long.deinit(alloc);
    try long.print(alloc, header, .{ "3333", "2026-11-01T11:00:00Z" });
    try long.appendSlice(alloc, "{\"type\":\"user\",\"id\":1,\"parent\":null,\"time\":\"t\",\"text\":\"");
    try long.appendNTimes(alloc, 'x', 70 * 1024);
    try long.appendSlice(alloc, "\"}\n");
    var middle: std.ArrayList(u8) = .empty;
    defer middle.deinit(alloc);
    try middle.print(alloc, header, .{ "2222", "2026-11-01T10:00:00Z" });
    try middle.appendSlice(alloc, user);
    const files = [_]struct { name: []const u8, data: []const u8 }{
        .{ .name = "20261101T090000Z_1111.jsonl", .data = big.items },
        .{ .name = "20261101T110000Z_3333.jsonl", .data = long.items },
        .{ .name = "20261101T100000Z_2222.jsonl", .data = middle.items },
        .{ .name = "20261101T120000Z_dead.jsonl", .data = "{\"type\":\"session\",\"version\":1,\"id\":\"dead\"" },
        .{ .name = "20261101T130000Z_beef.jsonl", .data = "not json\n" ++ user },
    };
    for (files) |file| {
        const rel = try std.fmt.allocPrint(alloc, "{s}/{s}", .{ dir, file.name });
        defer alloc.free(rel);
        try tmp.dir.writeFile(io, .{ .sub_path = rel, .data = file.data });
    }

    const items = try list(alloc, io, root, cwd);
    defer freeList(alloc, items);
    try testing.expectEqual(@as(usize, 3), items.len);
    try testing.expectEqualStrings("3333", items[0].id);
    try testing.expectEqualStrings("", items[0].first_prompt);
    try testing.expectEqualStrings("2222", items[1].id);
    try testing.expectEqualStrings("fix the parser", items[1].first_prompt);
    try testing.expectEqual(@as(u64, middle.items.len), items[1].bytes);
    try testing.expectEqualStrings("1111", items[2].id);
    try testing.expectEqualStrings("fix the parser", items[2].first_prompt);
    try testing.expectEqual(@as(u64, big.items.len), items[2].bytes);
    // `latest` and an id prefix see the big session too.
    const newest = (try find(alloc, io, root, cwd, latest)).?;
    defer alloc.free(newest);
    try testing.expect(std.mem.endsWith(u8, newest, "_3333.jsonl"));
    const one = try match(alloc, io, root, cwd, "11");
    defer one.deinit(alloc);
    try testing.expectEqualStrings("1111", one.one.id);
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
}

test {
    _ = inference;
}

test "fuzz session: any file loads, rebuilds, and exports, or is a typed error" {
    const fuzz = inference.fuzz;
    const Case = struct {
        fn one(_: void, smith: *fuzz.Smith) !void {
            var buf: [fuzz.max_input]u8 = undefined;
            const source = fuzz.bytes(smith, &buf);
            if (session.parseHeader(testing.allocator, source)) |header| header.deinit(testing.allocator) else |_| {}
            const loaded = session.parseText(testing.allocator, source, null) catch return;
            defer loaded.deinit();
            _ = lastBoundary(loaded);
            if (conversation(testing.allocator, loaded)) |c| c.deinit(testing.allocator) else |_| {}
            var discard: std.Io.Writer.Discarding = .init(&.{});
            session.exportMarkdown(loaded, &discard.writer) catch {};
        }
    };
    const header = "{\"type\":\"session\",\"version\":1,\"id\":\"a\",\"time\":\"t\",\"cwd\":\"/w\",\"model\":{\"path\":\"/m\"},\"effort\":\"off\",\"ctx_size\":8}\n";
    try testing.fuzz({}, Case.one, .{ .corpus = fuzz.corpus(&.{
        header ++ "{\"type\":\"user\",\"id\":1,\"parent\":null,\"time\":\"t\",\"text\":\"hi\"}\n" ++
            "{\"type\":\"assistant\",\"id\":2,\"parent\":1,\"time\":\"t\",\"text\":\"ok\",\"tool_calls\":[{\"id\":1,\"name\":\"read_file\",\"arguments\":\"{}\"}]}\n" ++
            "{\"type\":\"tool\",\"id\":3,\"parent\":2,\"time\":\"t\",\"tool_call_id\":1,\"text\":\"x\"}\n" ++
            "{\"type\":\"compaction\",\"id\":4,\"parent\":3,\"time\":\"t\",\"first_kept\":2,\"reason\":\"context_full\"}\n",
        header ++ "{\"type\":\"user\",\"id\":1,\"parent\":4294967296,\"time\":\"t\",\"text\":\"x\"}\n{}\n",
        header ++ "{\"type\":\"user\",\"id\":1,\"parent\":null,\"time\":\"t\",\"text\":\"x\",\"images\":[{\"path\":\"p\",\"width\":1099511627776}]}\n{",
        header ++ "{\"type\":\"compaction\",\"id\":1,\"parent\":null,\"time\":\"t\",\"first_kept\":99,\"reason\":\"x\"}\n",
    }) });
}

test "a resume replays the cheap cuts, starts at the latest hand-off, and takes the latest system block" {
    const alloc = std.testing.allocator;
    var buffer: std.Io.Writer.Allocating = .init(alloc);
    defer buffer.deinit();
    const w = &buffer.writer;
    try w.writeAll("{\"type\":\"session\",\"version\":2,\"id\":\"a\",\"time\":\"t\",\"cwd\":\"/w\",\"model\":{\"path\":\"/m\"},\"effort\":\"low\",\"ctx_size\":8}\n");
    const big = "{\"path\":\"a.txt\",\"content\":\"" ++ @as([600]u8, @splat('x')) ++ "\"}";
    const entries = [_]session.Entry{
        .{ .system = .{ .text = "S1" } },
        .{ .user = .{ .text = "go" } },
        .{ .assistant = .{ .thinking = "plan", .answer = "", .tool_calls = &.{.{ .id = 1, .name = "write_file", .arguments = big }}, .stop = "eos" } },
        .{ .tool_result = .{ .call = "1", .text = "created a.txt" } },
        .{ .compaction = .{ .first_kept = 2, .reason = loop.reasoning_dropped } },
        .{ .compaction = .{ .first_kept = 2, .reason = loop.arguments_stubbed } },
    };
    for (entries, 1..) |entry, id| try session.writeEntry(w, entry, @intCast(id), if (id == 1) null else @intCast(id - 1), "t");
    {
        const loaded = try session.parseText(alloc, buffer.written(), null);
        defer loaded.deinit();
        const c = try conversation(alloc, loaded);
        defer c.deinit(alloc);
        try std.testing.expectEqual(@as(usize, 3), c.messages.len);
        try std.testing.expectEqualStrings("S1", c.system.?);
        try std.testing.expectEqualStrings("", c.messages[1].reasoning_content);
        try std.testing.expect(std.mem.indexOf(u8, c.messages[1].tool_calls[0].arguments, handoff.stub_marker) != null);
    }
    const later = [_]session.Entry{
        .{ .handoff = .{ .text = "H1" } },
        .{ .system = .{ .text = "S2" } },
        .{ .user = .{ .text = "more" } },
        .{ .handoff = .{ .text = "H2" } },
        .{ .assistant = .{ .answer = "ok", .stop = "eos" } },
    };
    for (later, entries.len + 1..) |entry, id| try session.writeEntry(w, entry, @intCast(id), @intCast(id - 1), "t");
    const loaded = try session.parseText(alloc, buffer.written(), null);
    defer loaded.deinit();
    const c = try conversation(alloc, loaded);
    defer c.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 2), c.messages.len);
    try std.testing.expectEqualStrings("H2", c.messages[0].content);
    try std.testing.expectEqualStrings("ok", c.messages[1].content);
    try std.testing.expectEqualStrings("S2", c.system.?);
}
