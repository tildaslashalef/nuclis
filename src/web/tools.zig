//! The two web tools as the model sees them, one definition for the agent
//! and for `nuclis serve`'s hosted tools: names, descriptions, parameter
//! schemas, argument checks, and the text a result or a failure becomes.
//! Every limit here is a host constant; the model chooses only the query,
//! the URL, and the offset. A failure is a rendered result with `is_error`,
//! never a Zig error (only allocation failure is).
const std = @import("std");
const client = @import("client.zig");
const Allocator = std.mem.Allocator;

pub const Definition = struct {
    name: []const u8,
    description: []const u8,
    /// A JSON object: the parameter schema the model is shown.
    parameters: []const u8,
};

pub const search: Definition = .{
    .name = "web_search",
    .description = "Search the web through the user's search service: at most 8 results, each a title, a URL, and a snippet. For documentation, APIs, error messages, and anything newer than what you know.",
    .parameters = "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}},\"required\":[\"query\"]}",
};

pub const fetch: Definition = .{
    .name = "web_fetch",
    .description = "Fetch one public web page as Markdown text, 16000 characters at a time; offset (in characters) continues a long page.",
    .parameters = "{\"type\":\"object\",\"properties\":{\"url\":{\"type\":\"string\"},\"offset\":{\"type\":\"integer\"}},\"required\":[\"url\"]}",
};

pub const all = [_]Definition{ search, fetch };

pub const max_results: u8 = 8;
/// About 4K tokens of text per call; the loop's own cut still applies.
pub const page_chars: u32 = 16_000;
pub const max_query_bytes = 400;
pub const max_url_bytes = 2048;

pub fn isWebTool(name: []const u8) bool {
    return std.mem.eql(u8, name, search.name) or std.mem.eql(u8, name, fetch.name);
}

/// What the model reads, and the transcript's one-line summary. Owned.
pub const Rendered = struct {
    text: []u8,
    summary: ?[]u8 = null,
    is_error: bool = false,
    truncated: bool = false,

    pub fn deinit(self: *Rendered, alloc: Allocator) void {
        alloc.free(self.text);
        if (self.summary) |s| alloc.free(s);
        self.* = undefined;
    }
};

/// Runs `name` (`web_search` or `web_fetch`) with the model's JSON
/// `arguments`; null for any other name.
pub fn run(service: *client.Service, alloc: Allocator, name: []const u8, arguments: []const u8, beat: ?client.Beat) Allocator.Error!?Rendered {
    if (std.mem.eql(u8, name, search.name)) return try runSearch(service, alloc, arguments, beat);
    if (std.mem.eql(u8, name, fetch.name)) return try runFetch(service, alloc, arguments, beat);
    return null;
}

pub fn runSearch(service: *client.Service, alloc: Allocator, arguments: []const u8, beat: ?client.Beat) Allocator.Error!Rendered {
    const Args = struct { query: []const u8 };
    const parsed = std.json.parseFromSlice(Args, alloc, arguments, .{ .ignore_unknown_fields = true }) catch
        return fail(alloc, "web_search: arguments must be a JSON object with a \"query\" string", .{});
    defer parsed.deinit();
    const query = std.mem.trim(u8, parsed.value.query, " \t\r\n");
    if (query.len == 0) return fail(alloc, "web_search: query is empty", .{});
    if (query.len > max_query_bytes)
        return fail(alloc, "web_search: the query is longer than {d} bytes; search for the concept or the error message, not pasted text", .{max_query_bytes});
    var reply = try service.search(alloc, query, max_results, beat);
    defer reply.deinit();
    return switch (reply.outcome) {
        .ok => |found| renderSearch(alloc, found),
        .failed => |f| failure(alloc, "web_search", f),
    };
}

pub fn runFetch(service: *client.Service, alloc: Allocator, arguments: []const u8, beat: ?client.Beat) Allocator.Error!Rendered {
    const Args = struct { url: []const u8, offset: ?u64 = null };
    const parsed = std.json.parseFromSlice(Args, alloc, arguments, .{ .ignore_unknown_fields = true }) catch
        return fail(alloc, "web_fetch: arguments must be a JSON object with a \"url\" string and an optional non-negative integer \"offset\"", .{});
    defer parsed.deinit();
    const url = std.mem.trim(u8, parsed.value.url, " \t\r\n");
    if (checkUrl(url)) |problem| return fail(alloc, "web_fetch: {s}", .{problem});
    const offset = parsed.value.offset orelse 0;
    var reply = try service.fetch(alloc, url, offset, page_chars, beat);
    defer reply.deinit();
    return switch (reply.outcome) {
        .ok => |page| renderFetch(alloc, page, offset),
        .failed => |f| failure(alloc, "web_fetch", f),
    };
}

/// Why `url` never leaves the machine, or null when it may.
fn checkUrl(url: []const u8) ?[]const u8 {
    if (url.len == 0) return "url is empty";
    if (url.len > max_url_bytes) return "the url is longer than 2048 bytes";
    const uri = std.Uri.parse(url) catch return "the url is not a valid URL";
    if (!std.ascii.eqlIgnoreCase(uri.scheme, "http") and !std.ascii.eqlIgnoreCase(uri.scheme, "https")) return "only http and https URLs are fetched";
    if (uri.host == null) return "the url has no host";
    return null;
}

fn renderSearch(alloc: Allocator, found: client.Search) Allocator.Error!Rendered {
    var text: std.ArrayList(u8) = .empty;
    errdefer text.deinit(alloc);
    const shown = found.results[0..@min(found.results.len, max_results)];
    if (shown.len == 0) try text.print(alloc, "no results for \"{s}\"", .{found.query});
    for (shown, 1..) |result, n| {
        try text.print(alloc, "{d}. {s} — {s}", .{ n, if (result.title.len > 0) result.title else result.url, result.url });
        if (result.published) |date| try text.print(alloc, " ({s})", .{date});
        try text.append(alloc, '\n');
        if (result.snippet.len > 0) try text.print(alloc, "   {s}\n", .{result.snippet});
    }
    if (found.engines_failed.len > 0) {
        if (text.items.len > 0 and text.items[text.items.len - 1] != '\n') try text.append(alloc, '\n');
        try text.appendSlice(alloc, "[engines failed: ");
        for (found.engines_failed, 0..) |engine, i| try text.print(alloc, "{s}{s}", .{ if (i > 0) ", " else "", engine });
        try text.appendSlice(alloc, "; results may be partial]");
    }
    const trimmed = std.mem.trimEnd(u8, text.items, "\n");
    text.shrinkRetainingCapacity(trimmed.len);
    var summary: std.ArrayList(u8) = .empty;
    errdefer summary.deinit(alloc);
    if (shown.len == 0) try summary.appendSlice(alloc, "no results") else try summary.print(alloc, "{d} result{s}", .{ shown.len, if (shown.len == 1) "" else "s" });
    if (found.engines_failed.len > 0) try summary.print(alloc, " · {d} engine{s} failed", .{ found.engines_failed.len, if (found.engines_failed.len == 1) "" else "s" });
    const owned_summary = try summary.toOwnedSlice(alloc);
    errdefer alloc.free(owned_summary);
    return .{ .text = try text.toOwnedSlice(alloc), .summary = owned_summary, .truncated = found.engines_failed.len > 0 };
}

fn renderFetch(alloc: Allocator, page: client.Fetched, offset: u64) Allocator.Error!Rendered {
    if (page.text.len == 0 and page.chars > 0 and offset >= page.chars)
        return fail(alloc, "web_fetch: offset {d} is past the end ({d} characters)", .{ offset, page.chars });
    const taken: u64 = std.unicode.utf8CountCodepoints(page.text) catch page.text.len;
    const end = offset + taken;
    var note: std.ArrayList(u8) = .empty;
    defer note.deinit(alloc);
    if (page.chars == 0) {
        try note.appendSlice(alloc, "no text extracted; a page that renders with JavaScript yields little, so fetch its source instead");
    } else {
        try note.print(alloc, "characters {d} to {d} of {d}", .{ offset, end, page.chars });
        if (end < page.chars) try note.print(alloc, " · truncated, continue with offset={d}", .{end});
        if (page.truncated and end >= page.chars) try note.appendSlice(alloc, " · the page was cut at the service's 5 MiB read; this is all it read");
    }
    var text: std.ArrayList(u8) = .empty;
    errdefer text.deinit(alloc);
    try text.appendSlice(alloc, if (page.final_url.len > 0) page.final_url else page.url);
    if (page.title.len > 0) try text.print(alloc, " — {s}", .{page.title});
    try text.append(alloc, '\n');
    if (page.text.len > 0) try text.print(alloc, "\n{s}\n", .{std.mem.trimEnd(u8, page.text, "\n")});
    try text.print(alloc, "\n[{s}]", .{note.items});
    const summary = try alloc.dupe(u8, note.items[0 .. std.mem.indexOf(u8, note.items, " · ") orelse note.items.len]);
    errdefer alloc.free(summary);
    return .{ .text = try text.toOwnedSlice(alloc), .summary = summary, .truncated = end < page.chars or page.truncated };
}

/// A failure the model reads: the tool, then one sentence naming the cause.
fn failure(alloc: Allocator, tool: []const u8, f: client.Failure) Allocator.Error!Rendered {
    const reason: []const u8 = if (std.mem.eql(u8, f.code, "busy"))
        "the search service is busy, try again"
    else if (std.mem.eql(u8, f.code, "cancelled"))
        "cancelled"
    else
        f.message;
    return fail(alloc, "{s}: {s}", .{ tool, if (reason.len > 0) reason else f.code });
}

fn fail(alloc: Allocator, comptime format: []const u8, args: anytype) Allocator.Error!Rendered {
    return .{ .text = try std.fmt.allocPrint(alloc, format, args), .is_error = true };
}

test "a search renders numbered results, dates, and the engines that failed" {
    const a = std.testing.allocator;
    const results = [_]client.Search.Result{
        .{ .title = "Zig Language Reference", .url = "https://ziglang.org/documentation/master/", .snippet = "Zig is a general-purpose language.", .published = "2026-09-01" },
        .{ .title = "", .url = "https://github.com/ziglang/zig" },
    };
    var r = try renderSearch(a, .{ .query = "zig", .results = &results, .engines_failed = &.{"mojeek"} });
    defer r.deinit(a);
    try std.testing.expectEqualStrings(
        \\1. Zig Language Reference — https://ziglang.org/documentation/master/ (2026-09-01)
        \\   Zig is a general-purpose language.
        \\2. https://github.com/ziglang/zig — https://github.com/ziglang/zig
        \\[engines failed: mojeek; results may be partial]
    , r.text);
    try std.testing.expectEqualStrings("2 results · 1 engine failed", r.summary.?);
    try std.testing.expect(!r.is_error);

    var none = try renderSearch(a, .{ .query = "qqq" });
    defer none.deinit(a);
    try std.testing.expectEqualStrings("no results for \"qqq\"", none.text);
    try std.testing.expectEqualStrings("no results", none.summary.?);
}

test "a fetched page states its range and how to continue" {
    const a = std.testing.allocator;
    var partial = try renderFetch(a, .{ .url = "https://e.org/a", .final_url = "https://e.org/a/", .title = "A", .text = "# A\n\nbé", .chars = 100, .truncated = true }, 10);
    defer partial.deinit(a);
    try std.testing.expectEqualStrings("https://e.org/a/ — A\n\n# A\n\nbé\n\n[characters 10 to 17 of 100 · truncated, continue with offset=17]", partial.text);
    try std.testing.expectEqualStrings("characters 10 to 17 of 100", partial.summary.?);
    try std.testing.expect(partial.truncated);

    var whole = try renderFetch(a, .{ .url = "https://e.org/t", .text = "plain", .chars = 5 }, 0);
    defer whole.deinit(a);
    try std.testing.expectEqualStrings("https://e.org/t\n\nplain\n\n[characters 0 to 5 of 5]", whole.text);

    var past = try renderFetch(a, .{ .url = "https://e.org/a", .chars = 100 }, 100);
    defer past.deinit(a);
    try std.testing.expect(past.is_error);
    try std.testing.expectEqualStrings("web_fetch: offset 100 is past the end (100 characters)", past.text);

    var empty = try renderFetch(a, .{ .url = "https://e.org/js" }, 0);
    defer empty.deinit(a);
    try std.testing.expect(std.mem.indexOf(u8, empty.text, "fetch its source instead") != null);
}

test "arguments are checked before anything leaves the machine" {
    const a = std.testing.allocator;
    // An unreachable base: a call that got past the checks would fail
    // differently (unreachable, not the argument's message).
    var service: client.Service = .init(a, std.testing.io, "http://127.0.0.1:9", "nuclis/test");
    defer service.deinit();
    const cases = [_]struct { name: []const u8, arguments: []const u8, needle: []const u8 }{
        .{ .name = "web_search", .arguments = "{}", .needle = "a \"query\" string" },
        .{ .name = "web_search", .arguments = "{\"query\": \"  \"}", .needle = "query is empty" },
        .{ .name = "web_search", .arguments = "{\"query\": \"" ++ @as([401]u8, @splat('x')) ++ "\"}", .needle = "longer than 400 bytes" },
        .{ .name = "web_fetch", .arguments = "{\"url\": \"file:///etc/passwd\"}", .needle = "only http and https" },
        .{ .name = "web_fetch", .arguments = "{\"url\": \"\"}", .needle = "url is empty" },
        .{ .name = "web_fetch", .arguments = "{\"url\": \"https://e.org\", \"offset\": -1}", .needle = "non-negative integer" },
        .{ .name = "web_fetch", .arguments = "{\"url\": \"https://e.org/" ++ @as([2048]u8, @splat('x')) ++ "\"}", .needle = "longer than 2048 bytes" },
    };
    for (cases) |case| {
        var r = (try run(&service, a, case.name, case.arguments, null)).?;
        defer r.deinit(a);
        try std.testing.expect(r.is_error);
        if (std.mem.indexOf(u8, r.text, case.needle) == null) {
            std.debug.print("expected \"{s}\" in \"{s}\"\n", .{ case.needle, r.text });
            return error.TestUnexpectedResult;
        }
    }
    try std.testing.expect((try run(&service, a, "bash", "{}", null)) == null);
}

test "the busy code reads as a retry, other failures as the service's message" {
    const a = std.testing.allocator;
    var busy = try failure(a, "web_search", .{ .code = "busy", .message = "every slot stayed taken for 2 s", .status = 503 });
    defer busy.deinit(a);
    try std.testing.expectEqualStrings("web_search: the search service is busy, try again", busy.text);
    var blocked = try failure(a, "web_fetch", .{ .code = "blocked_address", .message = "192.168.1.110 is not a public address", .status = 403 });
    defer blocked.deinit(a);
    try std.testing.expectEqualStrings("web_fetch: 192.168.1.110 is not a public address", blocked.text);
}
