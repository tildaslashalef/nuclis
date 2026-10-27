//! Hosted web exchanges kept between requests. A completion that ran hosted
//! web calls returns only its last assistant message; the client's next
//! request resends that, not the calls and results the model read before it.
//! The stash keeps what the model saw under a digest of the conversation as
//! the client sends it, through that message, and puts it back when the
//! message returns, so the model reads the results again and the completer
//! reuses its state. Bounded by count, bytes, and age; a miss is a request
//! without the exchange, as if there were no stash. Only the GPU worker
//! touches it. docs/guide/api.md § Hosted web tools.
const std = @import("std");
const inference = @import("inference");
const wire = @import("wire.zig");

const Allocator = std.mem.Allocator;
const Profile = inference.profiles;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const max_entries = 64;
pub const max_bytes = 8 << 20;
pub const max_age_s = 3600;

pub const Key = [Sha256.digest_length]u8;

/// A conversation as its client sends it, digested one message at a time:
/// role, trimmed content, images and clips, and an assistant's calls by name and
/// arguments. Reasoning and call ids are left out: clients drop the one and
/// rename the other.
pub const Chain = struct {
    hasher: Sha256 = .init(.{}),

    pub fn add(self: *Chain, message: Profile.Message, media: []const wire.Part) void {
        self.hasher.update(&.{@backingInt(message.role)});
        self.field(std.mem.trim(u8, message.content, " \t\r\n"));
        self.count(media.len);
        for (media) |part| {
            self.hasher.update(&.{@backingInt(part.kind)});
            self.field(part.bytes);
        }
        self.count(message.tool_calls.len);
        for (message.tool_calls) |call| {
            self.field(call.name);
            self.field(call.arguments);
        }
    }

    /// The digest of the messages added so far; the chain goes on.
    pub fn key(self: Chain) Key {
        var copy = self;
        var out: Key = undefined;
        copy.hasher.final(&out);
        return out;
    }

    fn field(self: *Chain, bytes: []const u8) void {
        self.count(bytes.len);
        self.hasher.update(bytes);
    }

    fn count(self: *Chain, n: usize) void {
        var bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &bytes, n, .little);
        self.hasher.update(&bytes);
    }
};

/// An exchange as stored: what the model saw from its first assistant
/// message on, call ids local (1, 2, …) and the client's own calls 0; the
/// hosted results of the last assistant message follow it.
const Entry = struct {
    key: Key,
    arena: std.heap.ArenaAllocator,
    messages: []const Profile.Message,
    bytes: usize,
    stored_s: i64,
    used: u64,
};

pub const Stash = struct {
    gpa: Allocator,
    entries: std.ArrayList(*Entry) = .empty,
    bytes: usize = 0,
    uses: u64 = 0,

    pub fn deinit(self: *Stash) void {
        for (self.entries.items) |entry| self.free(entry);
        self.entries.deinit(self.gpa);
    }

    /// Keeps a copy of `exchange` (local ids, as `localize` makes them) under
    /// `key`, making room by age, then least recent use. One larger than the
    /// whole stash is not kept.
    pub fn put(self: *Stash, key: Key, exchange: []const Profile.Message, now_s: i64) Allocator.Error!void {
        const size = sizeOf(exchange);
        if (size > max_bytes) return;
        if (self.find(key)) |i| self.drop(i);
        var i: usize = 0;
        while (i < self.entries.items.len) {
            if (now_s - self.entries.items[i].stored_s > max_age_s) self.drop(i) else i += 1;
        }
        while (self.entries.items.len >= max_entries or self.bytes + size > max_bytes) {
            var oldest: usize = 0;
            for (self.entries.items, 0..) |entry, j| if (entry.used < self.entries.items[oldest].used) {
                oldest = j;
            };
            self.drop(oldest);
        }
        const entry = try self.gpa.create(Entry);
        entry.* = .{ .key = key, .arena = .init(self.gpa), .messages = &.{}, .bytes = size, .stored_s = now_s, .used = 0 };
        errdefer self.free(entry);
        entry.messages = try copyMessages(entry.arena.allocator(), exchange);
        try self.entries.append(self.gpa, entry);
        self.bytes += size;
        self.uses += 1;
        entry.used = self.uses;
    }

    /// The exchange kept under `key`, unless it has aged out. Valid until the
    /// next `put`.
    pub fn get(self: *Stash, key: Key, now_s: i64) ?[]const Profile.Message {
        const i = self.find(key) orelse return null;
        const entry = self.entries.items[i];
        if (now_s - entry.stored_s > max_age_s) {
            self.drop(i);
            return null;
        }
        self.uses += 1;
        entry.used = self.uses;
        return entry.messages;
    }

    fn find(self: *const Stash, key: Key) ?usize {
        for (self.entries.items, 0..) |entry, i| if (std.mem.eql(u8, &entry.key, &key)) return i;
        return null;
    }

    fn drop(self: *Stash, i: usize) void {
        const entry = self.entries.swapRemove(i);
        self.bytes -= entry.bytes;
        self.free(entry);
    }

    fn free(self: *Stash, entry: *Entry) void {
        entry.arena.deinit();
        self.gpa.destroy(entry);
    }
};

/// The bytes a profile counts against its input limit, plus a little per
/// message for the slices themselves.
fn sizeOf(messages: []const Profile.Message) usize {
    var total: usize = 0;
    for (messages) |message| total += 64 + countedBytes(message);
    return total;
}

fn countedBytes(message: Profile.Message) usize {
    var total = message.content.len + message.reasoning_content.len;
    for (message.tool_calls) |call| total += call.name.len + call.arguments.len;
    return total;
}

fn copyMessages(arena: Allocator, messages: []const Profile.Message) Allocator.Error![]const Profile.Message {
    const out = try arena.alloc(Profile.Message, messages.len);
    for (messages, out) |message, *copy| {
        const calls = try arena.alloc(Profile.ToolCall, message.tool_calls.len);
        for (message.tool_calls, calls) |call, *c| c.* = .{ .id = call.id, .name = try arena.dupe(u8, call.name), .arguments = try arena.dupe(u8, call.arguments) };
        copy.* = .{
            .role = message.role,
            .content = try arena.dupe(u8, message.content),
            .reasoning_content = try arena.dupe(u8, message.reasoning_content),
            .tool_calls = calls,
            .tool_call_id = message.tool_call_id,
        };
    }
    return out;
}

/// `exchange` with every call id renumbered 1, 2, … by first appearance,
/// the client's calls (id 0) kept 0, and its text made valid UTF-8 as the
/// response sent it. Lives in `arena`.
pub fn localize(arena: Allocator, exchange: []const Profile.Message) Allocator.Error![]Profile.Message {
    var seen: std.ArrayList(u32) = .empty;
    const out = try arena.alloc(Profile.Message, exchange.len);
    for (exchange, out) |message, *copy| {
        const calls = try arena.alloc(Profile.ToolCall, message.tool_calls.len);
        for (message.tool_calls, calls) |call, *c| c.* = .{
            .id = if (call.id == 0) 0 else try localId(arena, &seen, call.id),
            .name = try wire.validUtf8(arena, call.name),
            .arguments = try wire.validUtf8(arena, call.arguments),
        };
        copy.* = .{
            .role = message.role,
            .content = try wire.validUtf8(arena, message.content),
            .reasoning_content = try wire.validUtf8(arena, message.reasoning_content),
            .tool_calls = calls,
            .tool_call_id = if (message.tool_call_id) |id| try localId(arena, &seen, id) else null,
        };
    }
    return out;
}

fn localId(arena: Allocator, seen: *std.ArrayList(u32), id: u32) Allocator.Error!u32 {
    for (seen.items, 1..) |known, local| if (known == id) return @intCast(local);
    try seen.append(arena, id);
    return @intCast(seen.items.len);
}

/// Keeps `exchange` under the client's conversation (`chain`, from
/// `restore`) extended by `reply`, the assistant message the client
/// received: its whole text and its function calls.
pub fn record(stash: *Stash, arena: Allocator, chain: Chain, reply: Profile.Message, exchange: []const Profile.Message, now_s: i64) Allocator.Error!void {
    var extended = chain;
    extended.add(reply, &.{});
    try stash.put(extended.key(), try localize(arena, exchange), now_s);
}

pub const Restored = struct {
    /// The client's conversation, digested, for the key of what this
    /// request returns.
    chain: Chain,
    /// Exchanges put back.
    spliced: usize,
};

/// Puts back every exchange `stash` keeps for an assistant message
/// `request` resends, its hosted ids numbered past every id the
/// conversation uses. A splice that would break the profile's limits, or
/// whose client results do not answer its calls, is skipped.
pub fn restore(stash: *Stash, arena: Allocator, request: *wire.Request, now_s: i64) Allocator.Error!Restored {
    const messages = request.messages;
    const limits: Profile.Limits = .{};
    var next_id: u32 = 1;
    var total_bytes: usize = 0;
    for (request.tools) |tool| total_bytes += tool.name.len + tool.description.len + tool.parameters.len;
    for (messages) |message| {
        for (message.tool_calls) |call| next_id = @max(next_id, call.id + 1);
        if (message.tool_call_id) |id| next_id = @max(next_id, id + 1);
        total_bytes += countedBytes(message);
    }
    var chain: Chain = .{};
    var spliced: usize = 0;
    var out: std.ArrayList(Profile.Message) = .empty;
    var counts: std.ArrayList(usize) = .empty;
    var media_at: usize = 0;
    var i: usize = 0;
    while (i < messages.len) {
        const message = messages[i];
        const media = request.media[media_at..][0..request.media_counts[i]];
        media_at += media.len;
        chain.add(message, media);
        if (message.role == .assistant) if (stash.get(chain.key(), now_s)) |exchange| {
            var end = i + 1;
            while (end < messages.len and messages[end].role == .tool) end += 1;
            const results = messages[i + 1 .. end];
            const added_messages = exchange.len - 1;
            const added_bytes = countedBytesAll(exchange) -| countedBytes(message);
            if (fits(exchange, message, results) and out.items.len + (messages.len - i) + added_messages <= limits.messages and total_bytes + added_bytes <= limits.input_bytes) {
                const before = out.items.len;
                next_id = try splice(arena, &out, exchange, message, results, next_id);
                try counts.appendNTimes(arena, 0, out.items.len - before);
                for (results) |result| chain.add(result, &.{});
                total_bytes += added_bytes;
                spliced += 1;
                i = end;
                continue;
            }
        };
        try out.append(arena, message);
        try counts.append(arena, request.media_counts[i]);
        i += 1;
    }
    if (spliced > 0) {
        request.messages = out.items;
        request.media_counts = counts.items;
    }
    return .{ .chain = chain, .spliced = spliced };
}

fn countedBytesAll(messages: []const Profile.Message) usize {
    var total: usize = 0;
    for (messages) |message| total += countedBytes(message);
    return total;
}

/// Whether `exchange` can replace `message`: its last assistant message
/// holds as many client calls as `message`, and `results` answers each of
/// them once.
fn fits(exchange: []const Profile.Message, message: Profile.Message, results: []const Profile.Message) bool {
    const last = lastAssistant(exchange) orelse return false;
    var client: usize = 0;
    for (exchange[last].tool_calls) |call| {
        if (call.id == 0) client += 1;
    }
    if (client != message.tool_calls.len or results.len != client) return false;
    for (message.tool_calls) |call| if (resultFor(results, call.id) == null) return false;
    return true;
}

/// Appends `exchange` in place of `message` and its `results`: hosted ids
/// from `next_id` on, the client's calls with `message`'s ids, and the last
/// group's results in its calls' order. Returns the next free id.
fn splice(arena: Allocator, out: *std.ArrayList(Profile.Message), stored_exchange: []const Profile.Message, message: Profile.Message, results: []const Profile.Message, next_id: u32) Allocator.Error!u32 {
    // The request's copy: a later `put` may free the entry.
    const exchange = try copyMessages(arena, stored_exchange);
    const last = lastAssistant(exchange).?;
    var hosted: u32 = 0;
    var client: usize = 0;
    var final_calls: []const Profile.ToolCall = &.{};
    for (exchange[0 .. last + 1], 0..) |stored, j| {
        const calls = try arena.alloc(Profile.ToolCall, stored.tool_calls.len);
        for (stored.tool_calls, calls) |call, *c| {
            c.* = call;
            if (call.id == 0) {
                c.id = message.tool_calls[client].id;
                client += 1;
            } else {
                c.id = next_id + call.id - 1;
                hosted = @max(hosted, call.id);
            }
        }
        var copy = stored;
        copy.tool_calls = calls;
        if (stored.tool_call_id) |id| copy.tool_call_id = next_id + id - 1;
        try out.append(arena, copy);
        if (j == last) final_calls = calls;
    }
    // The last group's results, hosted and the client's, in call order.
    client = 0;
    for (exchange[last].tool_calls, final_calls) |stored, call| {
        if (stored.id == 0) {
            try out.append(arena, resultFor(results, call.id).?);
        } else {
            var result = resultFor(exchange[last + 1 ..], stored.id).?;
            result.tool_call_id = call.id;
            try out.append(arena, result);
        }
    }
    return next_id + hosted;
}

fn lastAssistant(messages: []const Profile.Message) ?usize {
    var i = messages.len;
    while (i > 0) {
        i -= 1;
        if (messages[i].role == .assistant) return i;
    }
    return null;
}

fn resultFor(results: []const Profile.Message, id: u32) ?Profile.Message {
    for (results) |result| if (result.tool_call_id == id) return result;
    return null;
}

const testing = std.testing;

fn parsed(arena: Allocator, body: []const u8) !wire.Request {
    var problem: wire.Problem = .{};
    return wire.parseWith(arena, body, &problem, .{ .web = true });
}

fn digest(messages: []const Profile.Message) Key {
    var chain: Chain = .{};
    for (messages) |message| chain.add(message, &.{});
    return chain.key();
}

test "the chain digests what clients resend: reasoning, call ids, and outer whitespace do not count" {
    const call = [_]Profile.ToolCall{.{ .id = 1, .name = "save", .arguments = "{\"a\":1}" }};
    const renamed = [_]Profile.ToolCall{.{ .id = 9, .name = "save", .arguments = "{\"a\":1}" }};
    const sent = [_]Profile.Message{ .{ .role = .user, .content = "q" }, .{ .role = .assistant, .content = "Answer.\n", .reasoning_content = "hm", .tool_calls = &call } };
    const resent = [_]Profile.Message{ .{ .role = .user, .content = "q" }, .{ .role = .assistant, .content = "Answer.", .tool_calls = &renamed } };
    try testing.expectEqualSlices(u8, &digest(&sent), &digest(&resent));
    const edited = [_]Profile.Message{ .{ .role = .user, .content = "q" }, .{ .role = .assistant, .content = "Answer!", .tool_calls = &renamed } };
    try testing.expect(!std.mem.eql(u8, &digest(&sent), &digest(&edited)));
    // The same message in another conversation is another key.
    const elsewhere = [_]Profile.Message{ .{ .role = .user, .content = "other" }, .{ .role = .assistant, .content = "Answer.", .tool_calls = &renamed } };
    try testing.expect(!std.mem.eql(u8, &digest(&sent), &digest(&elsewhere)));
    // Field boundaries are part of the digest.
    const split_a = [_]Profile.Message{ .{ .role = .user, .content = "ab" }, .{ .role = .user, .content = "c" } };
    const split_b = [_]Profile.Message{ .{ .role = .user, .content = "a" }, .{ .role = .user, .content = "bc" } };
    try testing.expect(!std.mem.eql(u8, &digest(&split_a), &digest(&split_b)));
}

test "the stash is bounded by count (least recent use goes first), age, and bytes" {
    var stash: Stash = .{ .gpa = testing.allocator };
    defer stash.deinit();
    const exchange = [_]Profile.Message{.{ .role = .assistant, .content = "x" }};
    // Printable bytes: the digest trims whitespace.
    var keys: [max_entries + 1]Key = undefined;
    for (&keys, 0..) |*key, i| key.* = digest(&.{.{ .role = .user, .content = &.{@intCast(i + 33)} }});
    for (keys[0..max_entries]) |key| try stash.put(key, &exchange, 0);
    // Touching the first keeps it; the second is then the least recently used.
    try testing.expect(stash.get(keys[0], 10) != null);
    try stash.put(keys[max_entries], &exchange, 10);
    try testing.expectEqual(@as(usize, max_entries), stash.entries.items.len);
    try testing.expect(stash.get(keys[0], 10) != null);
    try testing.expect(stash.get(keys[1], 10) == null);
    try testing.expectEqualStrings("x", stash.get(keys[max_entries], 10).?[0].content);
    // Past the age, an entry is gone.
    try testing.expect(stash.get(keys[0], max_age_s + 1) == null);
    // One exchange larger than the whole stash is not kept, and evicts nothing.
    const before = stash.entries.items.len;
    const huge = try testing.allocator.alloc(u8, max_bytes);
    defer testing.allocator.free(huge);
    try stash.put(keys[0], &.{.{ .role = .assistant, .content = huge }}, 20);
    try testing.expectEqual(before, stash.entries.items.len);
    try testing.expect(stash.bytes <= max_bytes);
}

test "an answer after hosted rounds comes back with the rounds the model read" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var stash: Stash = .{ .gpa = testing.allocator };
    defer stash.deinit();

    var first = try parsed(arena,
        \\{"messages": [{"role": "user", "content": "What is Zig?"}], "tools": [{"type": "web_search"}]}
    );
    const restored = try restore(&stash, arena, &first, 0);
    try testing.expectEqual(@as(usize, 0), restored.spliced);
    // What the model saw: a search round, its result, the answer. Ids as
    // the request numbered them.
    const search = [_]Profile.ToolCall{.{ .id = 4, .name = "web_search", .arguments = "{\"query\":\"zig\"}" }};
    const exchange = [_]Profile.Message{
        .{ .role = .assistant, .content = "Searching. ", .reasoning_content = "look it up", .tool_calls = &search },
        .{ .role = .tool, .content = "1. Zig — https://ziglang.org/", .tool_call_id = 4 },
        .{ .role = .assistant, .content = "A language.", .reasoning_content = "found it" },
    };
    try record(&stash, arena, restored.chain, .{ .role = .assistant, .content = "Searching. A language." }, &exchange, 0);

    var next = try parsed(arena,
        \\{"messages": [{"role": "user", "content": "What is Zig?"},
        \\  {"role": "assistant", "content": "Searching. A language."},
        \\  {"role": "user", "content": "Who made it?"}], "tools": [{"type": "web_search"}]}
    );
    const again = try restore(&stash, arena, &next, 5);
    try testing.expectEqual(@as(usize, 1), again.spliced);
    const m = next.messages;
    try testing.expectEqual(@as(usize, 5), m.len);
    try testing.expectEqual(@as(usize, 5), next.media_counts.len);
    try testing.expectEqualStrings("Searching. ", m[1].content);
    try testing.expectEqualStrings("look it up", m[1].reasoning_content);
    try testing.expectEqual(@as(u32, 1), m[1].tool_calls[0].id);
    try testing.expectEqual(@as(?u32, 1), m[2].tool_call_id);
    try testing.expectEqualStrings("A language.", m[3].content);
    try testing.expectEqualStrings("Who made it?", m[4].content);
    try Profile.validate(arena, m, next.tools, .{});
    // The key covers the client's conversation, not the spliced one.
    var client: Chain = .{};
    for ([_][]const u8{ "What is Zig?", "Searching. A language.", "Who made it?" }, [_]Profile.Role{ .user, .assistant, .user }) |text, role| client.add(.{ .role = role, .content = text }, &.{});
    try testing.expectEqualSlices(u8, &client.key(), &again.chain.key());

    // An edited answer, or another conversation, puts nothing back.
    var edited = try parsed(arena,
        \\{"messages": [{"role": "user", "content": "What is Zig?"},
        \\  {"role": "assistant", "content": "A language."},
        \\  {"role": "user", "content": "Who made it?"}]}
    );
    try testing.expectEqual(@as(usize, 0), (try restore(&stash, arena, &edited, 5)).spliced);
    try testing.expectEqual(@as(usize, 3), edited.messages.len);
}

test "function calls returned beside hosted ones come back in the model's order, with the client's ids and results" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var stash: Stash = .{ .gpa = testing.allocator };
    defer stash.deinit();

    var first = try parsed(arena,
        \\{"messages": [{"role": "user", "content": "Note the Zig homepage."}],
        \\ "tools": [{"type": "web_search"}, {"type": "function", "function": {"name": "save_note"}}]}
    );
    const restored = try restore(&stash, arena, &first, 0);
    // One round: a search, the client's save_note, a second search; the
    // searches ran, the note went back to the client.
    const calls = [_]Profile.ToolCall{
        .{ .id = 1, .name = "web_search", .arguments = "{\"query\":\"zig\"}" },
        .{ .id = 0, .name = "save_note", .arguments = "{\"text\":\"ziglang.org\"}" },
        .{ .id = 2, .name = "web_search", .arguments = "{\"query\":\"zig docs\"}" },
    };
    const exchange = [_]Profile.Message{
        .{ .role = .assistant, .content = "", .tool_calls = &calls },
        .{ .role = .tool, .content = "first", .tool_call_id = 1 },
        .{ .role = .tool, .content = "second", .tool_call_id = 2 },
    };
    const reply_calls = [_]Profile.ToolCall{.{ .id = 0, .name = "save_note", .arguments = "{\"text\":\"ziglang.org\"}" }};
    try record(&stash, arena, restored.chain, .{ .role = .assistant, .content = "", .tool_calls = &reply_calls }, &exchange, 0);

    var next = try parsed(arena,
        \\{"messages": [{"role": "user", "content": "Note the Zig homepage."},
        \\  {"role": "assistant", "content": null, "tool_calls": [{"id": "call_x", "type": "function", "function": {"name": "save_note", "arguments": "{\"text\":\"ziglang.org\"}"}}]},
        \\  {"role": "tool", "tool_call_id": "call_x", "content": "saved"}],
        \\ "tools": [{"type": "web_search"}, {"type": "function", "function": {"name": "save_note"}}]}
    );
    try testing.expectEqual(@as(usize, 1), (try restore(&stash, arena, &next, 1)).spliced);
    const m = next.messages;
    try testing.expectEqual(@as(usize, 5), m.len);
    // The client's call keeps its id (1); the searches take 2 and 3.
    try testing.expectEqual(@as(u32, 2), m[1].tool_calls[0].id);
    try testing.expectEqual(@as(u32, 1), m[1].tool_calls[1].id);
    try testing.expectEqual(@as(u32, 3), m[1].tool_calls[2].id);
    try testing.expectEqualStrings("first", m[2].content);
    try testing.expectEqualStrings("saved", m[3].content);
    try testing.expectEqual(@as(?u32, 1), m[3].tool_call_id);
    try testing.expectEqualStrings("second", m[4].content);
    try Profile.validate(arena, m, next.tools, .{});

    // A client that has not answered its call yet gets no splice.
    var unanswered = try parsed(arena,
        \\{"messages": [{"role": "user", "content": "Note the Zig homepage."},
        \\  {"role": "assistant", "content": null, "tool_calls": [{"id": "call_x", "type": "function", "function": {"name": "save_note", "arguments": "{\"text\":\"ziglang.org\"}"}}]},
        \\  {"role": "user", "content": "never mind"}]}
    );
    try testing.expectEqual(@as(usize, 0), (try restore(&stash, arena, &unanswered, 1)).spliced);
}

test "a splice past the profile's message limit is skipped" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var stash: Stash = .{ .gpa = testing.allocator };
    defer stash.deinit();
    const limits: Profile.Limits = .{};
    // A stored exchange of many rounds, and a conversation already at the limit.
    const exchange = try arena.alloc(Profile.Message, 7);
    for (exchange[0..6], 0..) |*message, i| message.* = if (i % 2 == 0)
        .{ .role = .assistant, .content = "", .tool_calls = try arena.dupe(Profile.ToolCall, &.{.{ .id = @intCast(i / 2 + 1), .name = "web_search", .arguments = "{}" }}) }
    else
        .{ .role = .tool, .content = "r", .tool_call_id = @intCast(i / 2 + 1) };
    exchange[6] = .{ .role = .assistant, .content = "done" };
    var chain: Chain = .{};
    chain.add(.{ .role = .user, .content = "q" }, &.{});
    try record(&stash, arena, chain, .{ .role = .assistant, .content = "done" }, exchange, 0);
    const counts = try arena.alloc(usize, limits.messages);
    @memset(counts, 0);
    var request: wire.Request = .{ .messages = try arena.alloc(Profile.Message, limits.messages), .media = &.{}, .media_counts = counts, .tools = &.{} };
    for (request.messages, 0..) |*message, i| message.* = switch (i) {
        0 => .{ .role = .user, .content = "q" },
        1 => .{ .role = .assistant, .content = "done" },
        else => .{ .role = if (i % 2 == 0) .user else .assistant, .content = "more" },
    };
    try testing.expectEqual(@as(usize, 0), (try restore(&stash, arena, &request, 0)).spliced);
    try testing.expectEqual(limits.messages, request.messages.len);
    // With room, the same conversation takes it.
    request.messages = request.messages[0..3];
    request.media_counts = request.media_counts[0..3];
    try testing.expectEqual(@as(usize, 1), (try restore(&stash, arena, &request, 0)).spliced);
    try testing.expectEqual(@as(usize, 9), request.messages.len);
}
