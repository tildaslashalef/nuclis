//! The session log: one append-only JSONL file per conversation under
//! `~/.nuclis/agent/sessions/`, and the markdown export derived from it.
//! A file, not a database: a conversation log needs none of a database's
//! properties, and this module states and enforces the ones it does need.
//!
//! - **Append-only.** One `write` per entry at the end of the file, with the
//!   header written lazily before the first one. A session that asks nothing
//!   writes no file — which is how print mode and the tests stay silent.
//! - **A truncated last line is dropped on load**; any other unparseable line
//!   is a typed error naming it. The last line is where a crash lands; earlier
//!   damage a reader should not paper over.
//! - **`version` is the migration key.** A newer file is rejected with
//!   `UnsupportedSessionVersion`, and nothing here rewrites a file it did not
//!   create.
//! - **Every entry carries `id` and `parent`**, even though this phase writes
//!   a linear chain, so branching and cancel-rewind need no format migration.
//! - **Entries store what the model saw and produced** — prompt, reasoning,
//!   answer, stop, measurements — never terminal styling. `/save` derives
//!   markdown from the same entries; there is no second transcript format.
//!
//! Serialization lives here rather than on `tui.event`, because an entry is
//! not an event: a turn is hundreds of `answer_delta` events and one
//! `assistant` entry. `Session` owns its path and header strings; `load`
//! returns everything behind a `Loaded` the caller deinits.
const std = @import("std");
const inference = @import("inference");
const paths = @import("../paths.zig");
const model = @import("../model.zig");

const Profile = inference.profiles;
const Allocator = std.mem.Allocator;

/// Bumped only when an existing key changes meaning or disappears; adding a
/// key is backwards compatible because a reader ignores what it does not know.
pub const format_version: u32 = 1;

/// A session file is a conversation, not a log of a machine: a bounded read
/// is still wise, and 64 MiB is far past any real one.
pub const max_file_bytes = 64 * 1024 * 1024;

/// Bytes of the working-directory slug before it is hashed instead.
pub const max_slug_bytes = 200;

pub const Header = struct {
    type: []const u8 = "session",
    version: u32 = format_version,
    /// A random 128-bit identifier in hex; also the name of the file.
    id: []const u8,
    /// RFC 3339 UTC, second resolution.
    time: []const u8,
    cwd: []const u8,
    model: Model,
    effort: []const u8,
    ctx_size: usize,

    pub const Model = struct { path: []const u8, sha256: ?[]const u8 = null };

    /// Frees a header whose strings were allocated by `parseHeader`. The header
    /// `create` builds points into its own `storage`; that one must not be
    /// passed here.
    pub fn deinit(self: Header, alloc: Allocator) void {
        alloc.free(self.id);
        alloc.free(self.time);
        alloc.free(self.cwd);
        alloc.free(self.model.path);
        if (self.model.sha256) |digest| alloc.free(digest);
        alloc.free(self.effort);
    }
};

/// Per-turn measurements, with `bench`'s definitions.
pub const Stats = struct {
    prompt_tokens: usize = 0,
    generated: usize = 0,
    prefill_seconds: f64 = 0,
    decode_seconds: f64 = 0,
    /// The step's reasoning time; 0 in files written before it was kept.
    thinking_seconds: f64 = 0,
    replayed: bool = false,
};

/// One line of the file after the header. The tag is the entry's `type`, and
/// the payload's fields are written beside `id`, `parent`, and `time`.
pub const Entry = union(enum) {
    /// `images` are the attachments by path and grid, never pixels: a
    /// resumed session decodes them again (absent in older files).
    user: struct { text: []const u8, images: []const ImageEntry = &.{} },
    assistant: struct {
        thinking: []const u8 = "",
        answer: []const u8 = "",
        /// The native calls the profile decoded, with the host correlation ids
        /// their results answer. Empty on a step that called no tool; the
        /// loader rebuilds the conversation from these and the `tool_result`
        /// entries.
        tool_calls: []const Profile.ToolCall = &.{},
        stop: []const u8,
        stats: Stats = .{},
    },
    tool_result: struct {
        call: []const u8,
        text: []const u8,
        truncated: bool = false,
        is_error: bool = false,
        /// The transcript's detail row; absent in files written before it
        /// existed, which load as empty.
        summary: []const u8 = "",
    },
    effort: struct { effort: []const u8 },
    context: struct { ctx_size: usize },
    compaction: struct { first_kept: usize, reason: []const u8 },
    notice: struct { text: []const u8 },
};

/// One attached image as the file records it.
pub const ImageEntry = struct {
    path: []const u8,
    width: u32 = 0,
    height: u32 = 0,
    width_tokens: u32 = 0,
    height_tokens: u32 = 0,
};

/// An entry with the chain fields a reader needs.
pub const Record = struct {
    id: u32,
    parent: ?u32,
    time: []const u8,
    entry: Entry,
};

// ----- writing -----

pub const Session = struct {
    alloc: Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    /// Where the file goes, or null for a session that writes nothing (no
    /// user root, print mode, `--no-session`). Owned.
    path: ?[]u8,
    header: Header,
    /// Owned copies of the header's strings, freed together.
    storage: []u8,
    next_id: u32 = 1,
    last_id: ?u32 = null,
    /// Set once the header has been written.
    started: bool = false,

    pub fn deinit(self: *Session) void {
        if (self.path) |p| self.alloc.free(p);
        self.alloc.free(self.storage);
        self.* = undefined;
    }

    /// Appends one entry, creating the directory and writing the header on
    /// the first call. A session with no path records nothing and reports
    /// success: the caller should not have to care whether logging is on.
    pub fn append(self: *Session, entry: Entry, now: []const u8) !void {
        const path = self.path orelse return;
        // Only create what is missing: `createDirPath` on a path that already
        // exists is not always a no-op (an absolute path through a symlinked
        // directory, `/tmp` on macOS, reports NotDir).
        if (std.fs.path.dirname(path)) |parent| {
            self.dir.access(self.io, parent, .{}) catch try self.dir.createDirPath(self.io, parent);
        }
        const file = try self.dir.createFile(self.io, path, .{ .truncate = false });
        defer file.close(self.io);
        const end = (try file.stat(self.io)).size;
        var buffer: [4096]u8 = undefined;
        var writer = file.writerStreaming(self.io, &buffer);
        try writer.seekTo(end);
        if (!self.started and end == 0) try writeHeader(&writer.interface, self.header);
        self.started = true;
        const id = self.next_id;
        try writeEntry(&writer.interface, entry, id, self.last_id, now);
        // One flush per entry: the file is complete at every turn boundary,
        // which is the whole crash story.
        try writer.interface.flush();
        self.next_id += 1;
        self.last_id = id;
    }
};

/// Opens (does not create) a session for the current working directory. The
/// file is named at this point but written only when something is appended.
/// `root_dir` null — no user root — yields a session that writes nothing.
pub fn create(alloc: Allocator, io: std.Io, dir: std.Io.Dir, options: Options) !Session {
    var storage: std.ArrayList(u8) = .empty;
    errdefer storage.deinit(alloc);
    const header: Header = .{
        .id = try keep(alloc, &storage, options.id),
        .time = try keep(alloc, &storage, options.time),
        .cwd = try keep(alloc, &storage, options.cwd),
        .model = .{
            .path = try keep(alloc, &storage, options.model_path),
            .sha256 = if (options.model_sha256) |digest| try keep(alloc, &storage, digest) else null,
        },
        .effort = try keep(alloc, &storage, options.effort),
        .ctx_size = options.ctx_size,
    };
    const owned = try storage.toOwnedSlice(alloc);
    errdefer alloc.free(owned);
    // The strings above point into `storage`'s buffer, which `toOwnedSlice`
    // may have shrunk in place; rebuild them against the final allocation so
    // nothing points at a freed byte.
    var rebuilt = header;
    var at: usize = 0;
    rebuilt.id = slice(owned, &at, options.id.len);
    rebuilt.time = slice(owned, &at, options.time.len);
    rebuilt.cwd = slice(owned, &at, options.cwd.len);
    rebuilt.model.path = slice(owned, &at, options.model_path.len);
    if (options.model_sha256) |digest| rebuilt.model.sha256 = slice(owned, &at, digest.len);
    rebuilt.effort = slice(owned, &at, options.effort.len);
    const path: ?[]u8 = if (options.override_path) |given|
        try alloc.dupe(u8, given)
    else if (options.root_dir) |root|
        try filePath(alloc, root, options.cwd, options.time, options.id)
    else
        null;
    return .{ .alloc = alloc, .io = io, .dir = dir, .path = path, .header = rebuilt, .storage = owned };
}

pub const Options = struct {
    /// The user root (`~/.nuclis`); null writes nothing.
    root_dir: ?[]const u8,
    /// An explicit file, given instead of the computed one (`--session` in
    /// print mode). It wins over `root_dir`.
    override_path: ?[]const u8 = null,
    id: []const u8,
    time: []const u8,
    cwd: []const u8,
    model_path: []const u8,
    model_sha256: ?[]const u8 = null,
    effort: []const u8,
    ctx_size: usize,
};

fn keep(alloc: Allocator, storage: *std.ArrayList(u8), bytes: []const u8) ![]const u8 {
    const start = storage.items.len;
    try storage.appendSlice(alloc, bytes);
    return storage.items[start..];
}

fn slice(owned: []const u8, at: *usize, len: usize) []const u8 {
    const start = at.*;
    at.* += len;
    return owned[start..at.*];
}

/// A random 128-bit identifier in lowercase hex; the session's name.
pub fn newId(io: std.Io, buffer: *[32]u8) []const u8 {
    var raw: [16]u8 = undefined;
    io.randomSecure(&raw) catch io.random(&raw);
    return std.fmt.bufPrint(buffer, "{x}", .{&raw}) catch unreachable;
}

/// `<root>/agent/sessions/<cwd-slug>/<stamp>_<id>.jsonl`. The slug keeps
/// sessions of different projects apart without nesting the filesystem's own
/// tree inside the agent's.
pub fn filePath(alloc: Allocator, root_dir: []const u8, cwd: []const u8, time: []const u8, id: []const u8) ![]u8 {
    const slug = try cwdSlug(alloc, cwd);
    defer alloc.free(slug);
    var stamp: [17]u8 = undefined;
    const name = try std.fmt.allocPrint(alloc, "{s}/{s}/{s}_{s}.jsonl", .{ paths.sessions_dir, slug, compactTime(&stamp, time), id });
    defer alloc.free(name);
    return paths.agentPath(alloc, root_dir, name);
}

/// The working directory as one path component: the leading separator
/// dropped, every other replaced by `-`. A very long path is truncated and
/// given a hash suffix, so two deep directories can never collide.
pub fn cwdSlug(alloc: Allocator, cwd: []const u8) ![]u8 {
    const trimmed = std.mem.trim(u8, cwd, "/");
    const flat = try alloc.dupe(u8, if (trimmed.len == 0) "root" else trimmed);
    for (flat) |*c| {
        if (c.* == '/' or c.* == '\\') c.* = '-';
    }
    if (flat.len <= max_slug_bytes) return flat;
    defer alloc.free(flat);
    return std.fmt.allocPrint(alloc, "{s}-{x}", .{ flat[0 .. max_slug_bytes - 17], std.hash.Wyhash.hash(0, cwd) });
}

/// `2026-09-12T10:00:00Z` as `20260912T100000Z`: the same instant, in a name
/// that sorts and needs no quoting.
pub fn compactTime(buffer: *[17]u8, time: []const u8) []const u8 {
    var at: usize = 0;
    for (time) |c| {
        if (c == '-' or c == ':') continue;
        if (at == buffer.len) break;
        buffer[at] = c;
        at += 1;
    }
    return buffer[0..at];
}

/// `<root>/agent/exports/<name>`: where `/save` writes without a path.
pub fn exportPath(alloc: Allocator, root_dir: []const u8, id: []const u8, time: []const u8) ![]u8 {
    var stamp: [17]u8 = undefined;
    const name = try std.fmt.allocPrint(alloc, "{s}/{s}_{s}.md", .{ paths.exports_dir, compactTime(&stamp, time), id });
    defer alloc.free(name);
    return paths.agentPath(alloc, root_dir, name);
}

pub fn writeHeader(out: *std.Io.Writer, header: Header) !void {
    try std.json.Stringify.value(header, .{ .emit_null_optional_fields = false }, out);
    try out.writeByte('\n');
}

/// One entry line: the chain fields, then the payload's own fields flattened
/// beside them.
///
/// Zig note: `inline else` gives the payload with its concrete type, so the
/// reflection loop below is resolved at compile time — adding a field to an
/// entry writes it with no edit here, and adding a *kind* cannot be forgotten.
pub fn writeEntry(out: *std.Io.Writer, entry: Entry, id: u32, parent: ?u32, time: []const u8) !void {
    var s: std.json.Stringify = .{ .writer = out };
    try s.beginObject();
    try s.objectField("type");
    try s.write(@tagName(entry));
    try s.objectField("id");
    try s.write(id);
    try s.objectField("parent");
    if (parent) |value| try s.write(value) else try s.write(null);
    try s.objectField("time");
    try s.write(time);
    switch (entry) {
        inline else => |payload| {
            inline for (@typeInfo(@TypeOf(payload)).@"struct".fields) |field| {
                const value = @field(payload, field.name);
                // A turn without images writes the line older readers know.
                const skip = comptime std.mem.eql(u8, field.name, "images");
                if (!skip or value.len != 0) {
                    try s.objectField(field.name);
                    try s.write(value);
                }
            }
        },
    }
    try s.endObject();
    try out.writeByte('\n');
}

// ----- reading -----

pub const Loaded = struct {
    arena: *std.heap.ArenaAllocator,
    header: Header,
    records: []const Record,

    pub fn deinit(self: Loaded) void {
        const child = self.arena.child_allocator;
        self.arena.deinit();
        child.destroy(self.arena);
    }
};

/// Which line a load failed on, for the message the caller prints.
pub const Diagnostic = struct { line: usize = 0 };

pub fn load(alloc: Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8, diagnostic: ?*Diagnostic) !Loaded {
    const arena = try alloc.create(std.heap.ArenaAllocator);
    errdefer alloc.destroy(arena);
    arena.* = .init(alloc);
    errdefer arena.deinit();
    const a = arena.allocator();
    const bytes = try dir.readFileAlloc(io, path, a, .limited(max_file_bytes));
    return parse(a, bytes, diagnostic, arena);
}

/// The same, from bytes: the form the round-trip test drives.
pub fn parseText(alloc: Allocator, source: []const u8, diagnostic: ?*Diagnostic) !Loaded {
    const arena = try alloc.create(std.heap.ArenaAllocator);
    errdefer alloc.destroy(arena);
    arena.* = .init(alloc);
    errdefer arena.deinit();
    const a = arena.allocator();
    return parse(a, try a.dupe(u8, source), diagnostic, arena);
}

/// The header line of a session file, without reading its entries: what a
/// session picker needs. Caller frees the strings with `Header.deinit`.
pub fn parseHeader(alloc: Allocator, source: []const u8) !Header {
    const end = std.mem.indexOfScalar(u8, source, '\n') orelse source.len;
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, source[0..end], .{});
    defer parsed.deinit();
    const object = switch (parsed.value) {
        .object => |object| object,
        else => return error.MalformedSessionLine,
    };
    return readHeader(alloc, object);
}

fn parse(a: Allocator, bytes: []const u8, diagnostic: ?*Diagnostic, arena: *std.heap.ArenaAllocator) !Loaded {
    var records: std.ArrayList(Record) = .empty;
    var header: ?Header = null;
    var lines = std.mem.splitScalar(u8, bytes, '\n');
    var line_number: usize = 0;
    while (lines.next()) |line| {
        line_number += 1;
        if (line.len == 0) continue;
        const last = lines.peek() == null;
        const value = std.json.parseFromSliceLeaky(std.json.Value, a, line, .{}) catch {
            // A crash lands in the last line and nowhere else.
            if (last) break;
            if (diagnostic) |d| d.line = line_number;
            return error.MalformedSessionLine;
        };
        const object = switch (value) {
            .object => |object| object,
            else => {
                if (last) break;
                if (diagnostic) |d| d.line = line_number;
                return error.MalformedSessionLine;
            },
        };
        if (header == null) {
            header = readHeader(a, object) catch |err| {
                if (diagnostic) |d| d.line = line_number;
                return err;
            };
            continue;
        }
        const record = readRecord(a, object) catch |err| {
            if (last and err == error.MalformedSessionLine) break;
            if (diagnostic) |d| d.line = line_number;
            return err;
        };
        try records.append(a, record);
    }
    return .{ .arena = arena, .header = header orelse return error.EmptySessionFile, .records = records.items };
}

fn readHeader(a: Allocator, object: std.json.ObjectMap) !Header {
    const kind = string(object.get("type")) orelse return error.MalformedSessionLine;
    if (!std.mem.eql(u8, kind, "session")) return error.MalformedSessionLine;
    const file_version = integer(object.get("version")) orelse return error.MalformedSessionLine;
    if (file_version > format_version) return error.UnsupportedSessionVersion;
    const model_object = switch (object.get("model") orelse .null) {
        .object => |m| m,
        else => return error.MalformedSessionLine,
    };
    return .{
        .version = @intCast(file_version),
        .id = try a.dupe(u8, string(object.get("id")) orelse return error.MalformedSessionLine),
        .time = try a.dupe(u8, string(object.get("time")) orelse ""),
        .cwd = try a.dupe(u8, string(object.get("cwd")) orelse ""),
        .model = .{
            .path = try a.dupe(u8, string(model_object.get("path")) orelse ""),
            .sha256 = if (string(model_object.get("sha256"))) |digest| try a.dupe(u8, digest) else null,
        },
        .effort = try a.dupe(u8, string(object.get("effort")) orelse "off"),
        .ctx_size = @intCast(integer(object.get("ctx_size")) orelse 0),
    };
}

fn readRecord(a: Allocator, object: std.json.ObjectMap) !Record {
    const kind = string(object.get("type")) orelse return error.MalformedSessionLine;
    const id = integer(object.get("id")) orelse return error.MalformedSessionLine;
    const parent: ?u32 = if (integer(object.get("parent"))) |value| @intCast(value) else null;
    const time = try a.dupe(u8, string(object.get("time")) orelse "");
    const entry: Entry = blk: {
        if (std.mem.eql(u8, kind, "user")) break :blk .{ .user = .{ .text = try text(a, object, "text"), .images = try readImages(a, object.get("images")) } };
        if (std.mem.eql(u8, kind, "assistant")) break :blk .{ .assistant = .{
            .thinking = try text(a, object, "thinking"),
            .answer = try text(a, object, "answer"),
            .tool_calls = try readCalls(a, object.get("tool_calls")),
            .stop = try text(a, object, "stop"),
            .stats = readStats(object.get("stats")),
        } };
        if (std.mem.eql(u8, kind, "tool_result")) break :blk .{ .tool_result = .{
            .call = try text(a, object, "call"),
            .text = try text(a, object, "text"),
            .truncated = boolean(object.get("truncated")),
            .is_error = boolean(object.get("is_error")),
            .summary = try text(a, object, "summary"),
        } };
        if (std.mem.eql(u8, kind, "effort")) break :blk .{ .effort = .{ .effort = try text(a, object, "effort") } };
        if (std.mem.eql(u8, kind, "context")) break :blk .{ .context = .{ .ctx_size = @intCast(integer(object.get("ctx_size")) orelse 0) } };
        if (std.mem.eql(u8, kind, "compaction")) break :blk .{ .compaction = .{
            .first_kept = @intCast(integer(object.get("first_kept")) orelse 0),
            .reason = try text(a, object, "reason"),
        } };
        if (std.mem.eql(u8, kind, "notice")) break :blk .{ .notice = .{ .text = try text(a, object, "text") } };
        return error.MalformedSessionLine;
    };
    return .{ .id = @intCast(id), .parent = parent, .time = time, .entry = entry };
}

fn readCalls(a: Allocator, value: ?std.json.Value) ![]const Profile.ToolCall {
    const array = switch (value orelse .null) {
        .array => |array| array,
        .null => return &.{},
        else => return error.MalformedSessionLine,
    };
    const calls = try a.alloc(Profile.ToolCall, array.items.len);
    for (array.items, 0..) |element, i| {
        const object = switch (element) {
            .object => |object| object,
            else => return error.MalformedSessionLine,
        };
        calls[i] = .{
            .id = @intCast(integer(object.get("id")) orelse 0),
            .name = try a.dupe(u8, string(object.get("name")) orelse ""),
            .arguments = try a.dupe(u8, string(object.get("arguments")) orelse ""),
        };
    }
    return calls;
}

fn readImages(a: Allocator, value: ?std.json.Value) ![]const ImageEntry {
    const array = switch (value orelse .null) {
        .array => |array| array,
        .null => return &.{},
        else => return error.MalformedSessionLine,
    };
    const images = try a.alloc(ImageEntry, array.items.len);
    for (array.items, 0..) |element, i| {
        const object = switch (element) {
            .object => |object| object,
            else => return error.MalformedSessionLine,
        };
        images[i] = .{
            .path = try a.dupe(u8, string(object.get("path")) orelse ""),
            .width = @intCast(integer(object.get("width")) orelse 0),
            .height = @intCast(integer(object.get("height")) orelse 0),
            .width_tokens = @intCast(integer(object.get("width_tokens")) orelse 0),
            .height_tokens = @intCast(integer(object.get("height_tokens")) orelse 0),
        };
    }
    return images;
}

fn readStats(value: ?std.json.Value) Stats {
    const object = switch (value orelse .null) {
        .object => |object| object,
        else => return .{},
    };
    return .{
        .prompt_tokens = @intCast(integer(object.get("prompt_tokens")) orelse 0),
        .generated = @intCast(integer(object.get("generated")) orelse 0),
        .prefill_seconds = number(object.get("prefill_seconds")),
        .decode_seconds = number(object.get("decode_seconds")),
        .thinking_seconds = number(object.get("thinking_seconds")),
        .replayed = boolean(object.get("replayed")),
    };
}

fn text(a: Allocator, object: std.json.ObjectMap, key: []const u8) ![]const u8 {
    return a.dupe(u8, string(object.get(key)) orelse "");
}

fn string(value: ?std.json.Value) ?[]const u8 {
    return switch (value orelse .null) {
        .string => |s| s,
        else => null,
    };
}

fn integer(value: ?std.json.Value) ?i64 {
    return switch (value orelse .null) {
        .integer => |i| i,
        else => null,
    };
}

fn number(value: ?std.json.Value) f64 {
    return switch (value orelse .null) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => 0,
    };
}

fn boolean(value: ?std.json.Value) bool {
    return switch (value orelse .null) {
        .bool => |b| b,
        else => false,
    };
}

// ----- markdown export -----

/// `/save`: the same entries as a document. Nothing is recomputed and nothing
/// is styled — a reader of the file and a reader of the terminal saw the same
/// conversation.
pub fn exportMarkdown(loaded: Loaded, out: *std.Io.Writer) !void {
    const header = loaded.header;
    try out.print("# nuclis session {s}\n\n", .{header.id});
    try out.print("- started: {s}\n- directory: `{s}`\n- model: `{s}`\n", .{ header.time, header.cwd, header.model.path });
    if (header.model.sha256) |digest| try out.print("- sha256: `{s}`\n", .{digest});
    try out.print("- context: {d}, effort: {s}\n\n", .{ header.ctx_size, header.effort });
    for (loaded.records) |record| {
        switch (record.entry) {
            .user => |u| {
                try out.print("---\n\n### You — {s}\n\n", .{record.time});
                try out.print("{s}\n\n", .{u.text});
                for (u.images, 1..) |image, n| try out.print("- image #{d}: `{s}` ({d}×{d} → {d}×{d} tokens)\n", .{ n, image.path, image.width, image.height, image.width_tokens, image.height_tokens });
                if (u.images.len > 0) try out.writeAll("\n");
            },
            .assistant => |assistant| {
                try out.print("### nuclis — {s}\n\n", .{record.time});
                if (assistant.thinking.len > 0) {
                    // Folded exactly as the transcript folds it.
                    try out.print("<details><summary>Thinking</summary>\n\n{s}\n\n</details>\n\n", .{assistant.thinking});
                }
                try out.print("{s}\n\n", .{assistant.answer});
                for (assistant.tool_calls) |call| {
                    try out.print("_tool call_ `{s}`:\n\n```json\n{s}\n```\n\n", .{ call.name, call.arguments });
                }
                if (!std.mem.eql(u8, assistant.stop, "eos")) try out.print("_stopped: {s}_\n\n", .{assistant.stop});
            },
            .tool_result => |result| {
                if (result.summary.len > 0) try out.print("_tool result {s} — {s}_\n", .{ result.call, result.summary }) else try out.print("_tool result {s}_\n", .{result.call});
                try out.print("```\n{s}\n```\n", .{result.text});
                if (result.truncated) try out.writeAll("_truncated_\n");
                try out.writeByte('\n');
            },
            .effort => |e| try out.print("_effort: {s}_\n\n", .{e.effort}),
            .context => |c| try out.print("_context: {d}_\n\n", .{c.ctx_size}),
            .compaction => |c| try out.print("_older turns dropped ({s}), from turn {d}_\n\n", .{ c.reason, c.first_kept }),
            .notice => |n| try out.print("_{s}_\n\n", .{n.text}),
        }
    }
}

// ----- tests -----

const testing = std.testing;

test "the header and the entries are the format the specification names" {
    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    try writeHeader(&buffer.writer, .{
        .id = "0123456789abcdef0123456789abcdef",
        .time = "2026-09-12T10:12:03Z",
        .cwd = "/Users/x/nuclis",
        .model = .{ .path = "/m.gguf", .sha256 = "90fd44e2" },
        .effort = "low",
        .ctx_size = 8192,
    });
    try testing.expectEqualStrings(
        "{\"type\":\"session\",\"version\":1,\"id\":\"0123456789abcdef0123456789abcdef\",\"time\":\"2026-09-12T10:12:03Z\"," ++
            "\"cwd\":\"/Users/x/nuclis\",\"model\":{\"path\":\"/m.gguf\",\"sha256\":\"90fd44e2\"},\"effort\":\"low\",\"ctx_size\":8192}\n",
        buffer.written(),
    );
    buffer.clearRetainingCapacity();
    try writeEntry(&buffer.writer, .{ .user = .{ .text = "hi" } }, 1, null, "2026-09-12T10:12:03Z");
    try testing.expectEqualStrings("{\"type\":\"user\",\"id\":1,\"parent\":null,\"time\":\"2026-09-12T10:12:03Z\",\"text\":\"hi\"}\n", buffer.written());
    buffer.clearRetainingCapacity();
    try writeEntry(&buffer.writer, .{ .compaction = .{ .first_kept = 2, .reason = "context_full" } }, 6, 5, "t");
    try testing.expectEqualStrings("{\"type\":\"compaction\",\"id\":6,\"parent\":5,\"time\":\"t\",\"first_kept\":2,\"reason\":\"context_full\"}\n", buffer.written());
}

test "a session round trips through the file, chained by id and parent" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const alloc = testing.allocator;
    var s = try create(alloc, testing.io, tmp.dir, .{
        .root_dir = "root",
        .id = "abc",
        .time = "2026-09-12T10:12:03Z",
        .cwd = "/w",
        .model_path = "/m.gguf",
        .model_sha256 = "deadbeef",
        .effort = "low",
        .ctx_size = 8192,
    });
    defer s.deinit();
    // Nothing is written before something is said.
    try testing.expectError(error.FileNotFound, tmp.dir.access(testing.io, s.path.?, .{}));
    try s.append(.{ .user = .{ .text = "why?" } }, "t1");
    try s.append(.{ .assistant = .{
        .thinking = "because",
        .answer = "42",
        .tool_calls = &.{.{ .id = 7, .name = "read_file", .arguments = "{\"path\":\"a.txt\"}" }},
        .stop = "eos",
        .stats = .{ .generated = 3, .decode_seconds = 0.25 },
    } }, "t2");
    try s.append(.{ .tool_result = .{ .call = "7", .text = "hello", .truncated = false, .is_error = false, .summary = "lines 1 to 1 of 1" } }, "t3");
    try s.append(.{ .effort = .{ .effort = "medium" } }, "t4");
    try s.append(.{ .notice = .{ .text = "older turns dropped" } }, "t5");

    const loaded = try load(alloc, testing.io, tmp.dir, s.path.?, null);
    defer loaded.deinit();
    try testing.expectEqualStrings("abc", loaded.header.id);
    try testing.expectEqual(format_version, loaded.header.version);
    try testing.expectEqualStrings("deadbeef", loaded.header.model.sha256.?);
    try testing.expectEqual(@as(usize, 8192), loaded.header.ctx_size);
    try testing.expectEqual(@as(usize, 5), loaded.records.len);
    try testing.expectEqualStrings("why?", loaded.records[0].entry.user.text);
    try testing.expect(loaded.records[0].parent == null);
    try testing.expectEqualStrings("because", loaded.records[1].entry.assistant.thinking);
    try testing.expectEqualStrings("42", loaded.records[1].entry.assistant.answer);
    try testing.expectEqual(@as(usize, 3), loaded.records[1].entry.assistant.stats.generated);
    try testing.expectEqual(@as(f64, 0.25), loaded.records[1].entry.assistant.stats.decode_seconds);
    // The decoded calls round trip with the host ids their results answer.
    try testing.expectEqual(@as(usize, 1), loaded.records[1].entry.assistant.tool_calls.len);
    try testing.expectEqual(@as(u32, 7), loaded.records[1].entry.assistant.tool_calls[0].id);
    try testing.expectEqualStrings("read_file", loaded.records[1].entry.assistant.tool_calls[0].name);
    try testing.expectEqualStrings("{\"path\":\"a.txt\"}", loaded.records[1].entry.assistant.tool_calls[0].arguments);
    try testing.expectEqualStrings("7", loaded.records[2].entry.tool_result.call);
    try testing.expectEqualStrings("hello", loaded.records[2].entry.tool_result.text);
    try testing.expectEqualStrings("lines 1 to 1 of 1", loaded.records[2].entry.tool_result.summary);
    // Every entry names the one before it, so a branch is representable later.
    for (loaded.records, 1..) |record, expected| {
        try testing.expectEqual(@as(u32, @intCast(expected)), record.id);
        if (expected > 1) try testing.expectEqual(@as(u32, @intCast(expected - 1)), record.parent.?);
    }
}

test "a truncated last line is dropped, a damaged earlier one is an error" {
    const alloc = testing.allocator;
    const good = "{\"type\":\"session\",\"version\":1,\"id\":\"a\",\"time\":\"t\",\"cwd\":\"/w\",\"model\":{\"path\":\"/m\"},\"effort\":\"off\",\"ctx_size\":8}\n" ++
        "{\"type\":\"user\",\"id\":1,\"parent\":null,\"time\":\"t\",\"text\":\"one\"}\n";
    const truncated = try parseText(alloc, good ++ "{\"type\":\"user\",\"id\":2,\"par", null);
    defer truncated.deinit();
    try testing.expectEqual(@as(usize, 1), truncated.records.len);

    var diagnostic: Diagnostic = .{};
    try testing.expectError(error.MalformedSessionLine, parseText(alloc, good ++ "not json\n{\"type\":\"user\",\"id\":3,\"parent\":2,\"time\":\"t\",\"text\":\"three\"}\n", &diagnostic));
    try testing.expectEqual(@as(usize, 3), diagnostic.line);
    // An unknown entry kind is damage too: the version is the migration key,
    // so a reader that does not know a type is reading a file it should not.
    try testing.expectError(error.MalformedSessionLine, parseText(alloc, good ++ "{\"type\":\"future\",\"id\":2,\"parent\":1,\"time\":\"t\"}\n{\"type\":\"notice\",\"id\":3,\"parent\":2,\"time\":\"t\",\"text\":\"x\"}\n", null));
}

test "a newer format is refused rather than guessed at" {
    const alloc = testing.allocator;
    try testing.expectError(error.UnsupportedSessionVersion, parseText(alloc, "{\"type\":\"session\",\"version\":2,\"id\":\"a\",\"time\":\"t\",\"cwd\":\"/w\",\"model\":{\"path\":\"/m\"},\"effort\":\"off\",\"ctx_size\":8}\n", null));
    try testing.expectError(error.MalformedSessionLine, parseText(alloc, "{\"type\":\"user\",\"id\":1}\n", null));
    try testing.expectError(error.EmptySessionFile, parseText(alloc, "\n", null));
}

test "the path names the working directory, the instant, and the session" {
    const alloc = testing.allocator;
    const path = try filePath(alloc, "/home/x/.nuclis", "/Users/alef/Code/nuclis", "2026-09-12T10:12:03Z", "abcdef");
    defer alloc.free(path);
    try testing.expectEqualStrings("/home/x/.nuclis/agent/sessions/Users-alef-Code-nuclis/20260912T101203Z_abcdef.jsonl", path);
    // A path too long for one component is truncated with a hash, so two deep
    // directories cannot collide.
    const deep = try alloc.alloc(u8, 400);
    defer alloc.free(deep);
    @memset(deep, 'd');
    deep[0] = '/';
    const slug = try cwdSlug(alloc, deep);
    defer alloc.free(slug);
    try testing.expect(slug.len <= max_slug_bytes);
    try testing.expect(std.mem.indexOfScalar(u8, slug, '/') == null);
    const root_slug = try cwdSlug(alloc, "/");
    defer alloc.free(root_slug);
    try testing.expectEqualStrings("root", root_slug);
    const exported = try exportPath(alloc, "/home/x/.nuclis", "abcdef", "2026-09-12T10:12:03Z");
    defer alloc.free(exported);
    try testing.expectEqualStrings("/home/x/.nuclis/agent/exports/20260912T101203Z_abcdef.md", exported);
}

test "the markdown export is derived from the entries, not from a second format" {
    const alloc = testing.allocator;
    const text_form = "{\"type\":\"session\",\"version\":1,\"id\":\"abc\",\"time\":\"2026-09-12T10:12:03Z\",\"cwd\":\"/w\",\"model\":{\"path\":\"/m.gguf\"},\"effort\":\"low\",\"ctx_size\":8192}\n" ++
        "{\"type\":\"user\",\"id\":1,\"parent\":null,\"time\":\"t1\",\"text\":\"why?\"}\n" ++
        "{\"type\":\"assistant\",\"id\":2,\"parent\":1,\"time\":\"t2\",\"thinking\":\"because\",\"answer\":\"**42**\"," ++
        "\"tool_calls\":[{\"id\":4,\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"a.txt\\\"}\"}],\"stop\":\"token_budget\",\"stats\":{\"generated\":3}}\n" ++
        "{\"type\":\"tool_result\",\"id\":3,\"parent\":2,\"time\":\"t3\",\"call\":\"4\",\"text\":\"hello\",\"truncated\":false,\"is_error\":false,\"summary\":\"lines 1 to 1 of 1\"}\n" ++
        "{\"type\":\"notice\",\"id\":4,\"parent\":3,\"time\":\"t4\",\"text\":\"older turns dropped\"}\n";
    const loaded = try parseText(alloc, text_form, null);
    defer loaded.deinit();
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    try exportMarkdown(loaded, &out.writer);
    const document = out.written();
    try testing.expect(std.mem.indexOf(u8, document, "# nuclis session abc") != null);
    try testing.expect(std.mem.indexOf(u8, document, "### You — t1") != null);
    try testing.expect(std.mem.indexOf(u8, document, "why?") != null);
    try testing.expect(std.mem.indexOf(u8, document, "<details><summary>Thinking</summary>") != null);
    // The answer is the model's markdown, kept as it was written.
    try testing.expect(std.mem.indexOf(u8, document, "**42**") != null);
    try testing.expect(std.mem.indexOf(u8, document, "_stopped: token_budget_") != null);
    // A tool exchange is part of the conversation: the call and its result are
    // both rendered, so the export reads like what the model saw.
    try testing.expect(std.mem.indexOf(u8, document, "_tool call_ `read_file`:") != null);
    try testing.expect(std.mem.indexOf(u8, document, "\"path\":\"a.txt\"") != null);
    try testing.expect(std.mem.indexOf(u8, document, "_tool result 4 — lines 1 to 1 of 1_") != null);
    try testing.expect(std.mem.indexOf(u8, document, "hello") != null);
    try testing.expect(std.mem.indexOf(u8, document, "_older turns dropped_") != null);
    // No escape sequence ever reaches the file.
    try testing.expect(std.mem.indexOfScalar(u8, document, 0x1b) == null);
}

test "an identifier is 128 random bits in hex" {
    var first: [32]u8 = undefined;
    var second: [32]u8 = undefined;
    const a = newId(testing.io, &first);
    const b = newId(testing.io, &second);
    try testing.expectEqual(@as(usize, 32), a.len);
    try testing.expect(!std.mem.eql(u8, a, b));
    for (a) |c| try testing.expect(std.ascii.isHex(c));
}

test {
    _ = model;
}

test "a user entry with images round-trips its paths and grids, and older files load without them" {
    const a = testing.allocator;
    var buffer: std.Io.Writer.Allocating = .init(a);
    defer buffer.deinit();
    const images = [_]ImageEntry{.{ .path = "/tmp/a.png", .width = 640, .height = 480, .width_tokens = 16, .height_tokens = 12 }};
    try writeEntry(&buffer.writer, .{ .user = .{ .text = "look [image #1]", .images = &images } }, 1, null, "t1");
    try testing.expect(std.mem.indexOf(u8, buffer.written(), "\"images\":[{\"path\":\"/tmp/a.png\",\"width\":640,\"height\":480,\"width_tokens\":16,\"height_tokens\":12}]") != null);
    var source: std.ArrayList(u8) = .empty;
    defer source.deinit(a);
    try source.appendSlice(a, "{\"type\":\"session\",\"version\":1,\"id\":\"a\",\"time\":\"t\",\"cwd\":\"/w\",\"model\":{\"path\":\"/m\"},\"effort\":\"off\",\"ctx_size\":8}\n");
    try source.appendSlice(a, buffer.written());
    try source.appendSlice(a, "{\"type\":\"user\",\"id\":2,\"parent\":1,\"time\":\"t2\",\"text\":\"older\"}\n");
    const loaded = try parseText(a, source.items, null);
    defer loaded.deinit();
    try testing.expectEqual(@as(usize, 1), loaded.records[0].entry.user.images.len);
    try testing.expectEqualStrings("/tmp/a.png", loaded.records[0].entry.user.images[0].path);
    try testing.expectEqual(@as(u32, 12), loaded.records[0].entry.user.images[0].height_tokens);
    try testing.expectEqual(@as(usize, 0), loaded.records[1].entry.user.images.len);
    var document: std.Io.Writer.Allocating = .init(a);
    defer document.deinit();
    try exportMarkdown(loaded, &document.writer);
    try testing.expect(std.mem.indexOf(u8, document.written(), "- image #1: `/tmp/a.png` (640×480 → 16×12 tokens)") != null);
}
