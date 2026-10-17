//! The web tools as the model sees them, one definition for the agent and
//! for `nuclis serve`'s hosted tools: names, descriptions, parameter
//! schemas, argument checks, and the text a result or a failure becomes.
//! Every limit here is a host constant; the model chooses only the query,
//! the freshness, the URL, and the offset. A failure is a rendered result with `is_error`,
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

const freshness_schema = "\"freshness\":{\"type\":\"string\",\"enum\":[\"day\",\"week\",\"month\",\"year\"]}";

pub const search: Definition = .{
    .name = "web_search",
    .description = "Search the web through the user's search service: at most 8 results, each a title, a URL, and a snippet. For documentation, APIs, error messages, and anything newer than what you know; freshness limits results to the last day, week, month, or year.",
    .parameters = "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}," ++ freshness_schema ++ "},\"required\":[\"query\"]}",
};

pub const fetch: Definition = .{
    .name = "web_fetch",
    .description = "Fetch one public web page as Markdown text, 16000 characters at a time; offset (in characters) continues a long page.",
    .parameters = "{\"type\":\"object\",\"properties\":{\"url\":{\"type\":\"string\"},\"offset\":{\"type\":\"integer\"}},\"required\":[\"url\"]}",
};

pub const context: Definition = .{
    .name = "web_context",
    .description = "Passages from several web pages that answer a question, each under its page's title and URL (about 2000 tokens in one call). Use it to answer a question from the web; use web_search to find pages or links, and web_fetch to read one page whole.",
    .parameters = "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}," ++ freshness_schema ++ "},\"required\":[\"query\"]}",
};

pub const all = [_]Definition{ search, fetch, context };

pub const max_results: u8 = 8;
/// About 4K tokens of text per call; the loop's own cut still applies.
pub const page_chars: u32 = 16_000;
pub const max_query_bytes = 400;
pub const max_url_bytes = 2048;
/// Results that show `extra_snippets`, and how many each.
pub const extra_results = 3;
pub const extras_per_result = 2;
/// About 3K tokens; the 2048-token budget came in under 9,200 characters
/// in the probe of ten questions.
pub const context_chars = 12_000;

pub fn isWebTool(name: []const u8) bool {
    for (all) |definition| if (std.mem.eql(u8, name, definition.name)) return true;
    return false;
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

/// Runs `name` (`web_search`, `web_fetch`, or `web_context`) with the
/// model's JSON `arguments`; null for any other name. `context_offered`:
/// the caller offers `web_context`, so a failure may point to it.
pub fn run(service: *client.Service, alloc: Allocator, name: []const u8, arguments: []const u8, context_offered: bool, beat: ?client.Beat) Allocator.Error!?Rendered {
    if (std.mem.eql(u8, name, search.name)) return try runSearch(service, alloc, arguments, beat);
    if (std.mem.eql(u8, name, fetch.name)) return try runFetch(service, alloc, arguments, context_offered, beat);
    if (std.mem.eql(u8, name, context.name)) return try runContext(service, alloc, arguments, beat);
    return null;
}

/// A query's arguments, checked: `query` and the optional `freshness`.
const Query = struct {
    parsed: std.json.Parsed(Args),
    query: []const u8,
    freshness: ?client.Freshness,

    const Args = struct { query: []const u8, freshness: ?[]const u8 = null };
};

/// The query's arguments, or the failed result the model reads.
fn readQuery(alloc: Allocator, tool: []const u8, arguments: []const u8) Allocator.Error!union(enum) { ok: Query, failed: Rendered } {
    const parsed = std.json.parseFromSlice(Query.Args, alloc, arguments, .{ .ignore_unknown_fields = true }) catch
        return .{ .failed = try fail(alloc, "{s}: arguments must be a JSON object with a \"query\" string and an optional \"freshness\" string", .{tool}) };
    errdefer parsed.deinit();
    const query = std.mem.trim(u8, parsed.value.query, " \t\r\n");
    const problem: ?Rendered = if (query.len == 0)
        try fail(alloc, "{s}: query is empty", .{tool})
    else if (query.len > max_query_bytes)
        try fail(alloc, "{s}: the query is longer than {d} bytes; search for the concept or the error message, not pasted text", .{ tool, max_query_bytes })
    else
        null;
    if (problem) |p| {
        parsed.deinit();
        return .{ .failed = p };
    }
    var freshness: ?client.Freshness = null;
    if (parsed.value.freshness) |text| {
        freshness = std.meta.stringToEnum(client.Freshness, text) orelse {
            const refused = try fail(alloc, "{s}: freshness must be day, week, month, or year", .{tool});
            parsed.deinit();
            return .{ .failed = refused };
        };
    }
    return .{ .ok = .{ .parsed = parsed, .query = query, .freshness = freshness } };
}

pub fn runSearch(service: *client.Service, alloc: Allocator, arguments: []const u8, beat: ?client.Beat) Allocator.Error!Rendered {
    const q = switch (try readQuery(alloc, search.name, arguments)) {
        .ok => |q| q,
        .failed => |r| return r,
    };
    defer q.parsed.deinit();
    var reply = try service.search(alloc, q.query, max_results, q.freshness, beat);
    defer reply.deinit();
    return switch (reply.outcome) {
        .ok => |found| renderSearch(alloc, found),
        .failed => |f| failure(alloc, search.name, f),
    };
}

pub fn runContext(service: *client.Service, alloc: Allocator, arguments: []const u8, beat: ?client.Beat) Allocator.Error!Rendered {
    const q = switch (try readQuery(alloc, context.name, arguments)) {
        .ok => |q| q,
        .failed => |r| return r,
    };
    defer q.parsed.deinit();
    var reply = try service.context(alloc, q.query, q.freshness, beat);
    defer reply.deinit();
    return switch (reply.outcome) {
        .ok => |passages| renderContext(alloc, passages),
        // A service older than `/v1/context` answers `not_found`.
        .failed => |f| failure(alloc, context.name, if (std.mem.eql(u8, f.code, "not_found")) .{ .code = "not_available", .message = f.message, .status = f.status } else f),
    };
}

pub fn runFetch(service: *client.Service, alloc: Allocator, arguments: []const u8, context_offered: bool, beat: ?client.Beat) Allocator.Error!Rendered {
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
        .failed => |f| if (std.mem.eql(u8, f.code, "upstream_failed") and std.mem.indexOf(u8, f.message, "bot challenge") != null)
            // holocron's own advice names its endpoint; the model reads ours.
            fail(alloc, "web_fetch: {s} is behind a bot challenge only a browser passes{s}", .{ hostOf(url), if (context_offered) "; web_context may hold its text" else "" })
        else
            failure(alloc, fetch.name, f),
    };
}

fn hostOf(url: []const u8) []const u8 {
    const uri = std.Uri.parse(url) catch return url;
    const host = uri.host orelse return url;
    return switch (host) {
        .raw, .percent_encoded => |h| h,
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
    if (found.altered_query) |altered| try text.print(alloc, "searched for \"{s}\" (the service corrected the query)\n", .{altered});
    if (shown.len == 0) try text.print(alloc, "no results for \"{s}\"", .{found.query});
    for (shown, 1..) |result, n| {
        try text.print(alloc, "{d}. {s} — {s}", .{ n, if (result.title.len > 0) result.title else result.url, result.url });
        if (result.published) |date| try text.print(alloc, " ({s})", .{date});
        try text.append(alloc, '\n');
        if (result.snippet.len > 0) try text.print(alloc, "   {s}\n", .{result.snippet});
        if (n <= extra_results) for (result.extra_snippets[0..@min(result.extra_snippets.len, extras_per_result)]) |extra|
            try text.print(alloc, "   {s}\n", .{extra});
    }
    try appendEnginesFailed(alloc, &text, found.engines_failed, "results");
    const trimmed = std.mem.trimEnd(u8, text.items, "\n");
    text.shrinkRetainingCapacity(trimmed.len);
    var summary: std.ArrayList(u8) = .empty;
    errdefer summary.deinit(alloc);
    if (shown.len == 0) try summary.appendSlice(alloc, "no results") else try summary.print(alloc, "{d} result{s}", .{ shown.len, if (shown.len == 1) "" else "s" });
    try appendFailedCount(alloc, &summary, found.engines_failed.len);
    const owned_summary = try summary.toOwnedSlice(alloc);
    errdefer alloc.free(owned_summary);
    return .{ .text = try text.toOwnedSlice(alloc), .summary = owned_summary, .truncated = found.engines_failed.len > 0 };
}

/// Sources numbered as search results, each snippet indented under its
/// source, cut at a source boundary past `context_chars`.
fn renderContext(alloc: Allocator, passages: client.Context) Allocator.Error!Rendered {
    var text: std.ArrayList(u8) = .empty;
    errdefer text.deinit(alloc);
    var block: std.ArrayList(u8) = .empty;
    defer block.deinit(alloc);
    if (passages.sources.len == 0) try text.appendSlice(alloc, "no passages matched the query");
    var chars: usize = 0;
    var shown: usize = 0;
    for (passages.sources, 1..) |source, n| {
        block.clearRetainingCapacity();
        if (n > 1) try block.append(alloc, '\n');
        try block.print(alloc, "{d}. {s} — {s}", .{ n, if (source.title.len > 0) source.title else source.url, source.url });
        if (source.published) |date| try block.print(alloc, " ({s})", .{date});
        try block.append(alloc, '\n');
        for (source.snippets) |snippet| {
            // A passage may span lines (a code example); each is indented.
            var lines = std.mem.splitScalar(u8, std.mem.trim(u8, snippet, " \t\r\n"), '\n');
            while (lines.next()) |line| try block.print(alloc, "   {s}\n", .{std.mem.trimEnd(u8, line, "\r")});
        }
        const block_chars = std.unicode.utf8CountCodepoints(block.items) catch block.items.len;
        if (shown > 0 and chars + block_chars > context_chars) break;
        if (block_chars > context_chars) {
            // One source alone past the bound: its first `context_chars`.
            const cut = codepointPrefix(block.items, context_chars);
            try text.appendSlice(alloc, cut);
            try text.appendSlice(alloc, "\n   [passage cut]\n");
            chars += context_chars;
        } else {
            try text.appendSlice(alloc, block.items);
            chars += block_chars;
        }
        shown += 1;
    }
    const hidden = passages.sources.len - shown;
    if (hidden > 0) try text.print(alloc, "\n[{d} more source{s} not shown]\n", .{ hidden, if (hidden == 1) "" else "s" });
    try appendEnginesFailed(alloc, &text, passages.engines_failed, "passages");
    const trimmed = std.mem.trimEnd(u8, text.items, "\n");
    text.shrinkRetainingCapacity(trimmed.len);
    var summary: std.ArrayList(u8) = .empty;
    errdefer summary.deinit(alloc);
    const count = passages.sources.len;
    if (count == 0) try summary.appendSlice(alloc, "no passages") else try summary.print(alloc, "{d} source{s}", .{ count, if (count == 1) "" else "s" });
    try appendFailedCount(alloc, &summary, passages.engines_failed.len);
    const owned_summary = try summary.toOwnedSlice(alloc);
    errdefer alloc.free(owned_summary);
    return .{ .text = try text.toOwnedSlice(alloc), .summary = owned_summary, .truncated = hidden > 0 or passages.engines_failed.len > 0 };
}

/// The longest prefix of `text` with at most `n` code points.
fn codepointPrefix(text: []const u8, n: usize) []const u8 {
    var i: usize = 0;
    var taken: usize = 0;
    while (i < text.len and taken < n) : (taken += 1) i += std.unicode.utf8ByteSequenceLength(text[i]) catch 1;
    return text[0..@min(i, text.len)];
}

fn appendEnginesFailed(alloc: Allocator, text: *std.ArrayList(u8), engines: []const []const u8, what: []const u8) Allocator.Error!void {
    if (engines.len == 0) return;
    if (text.items.len > 0 and text.items[text.items.len - 1] != '\n') try text.append(alloc, '\n');
    try text.appendSlice(alloc, "[engines failed: ");
    for (engines, 0..) |engine, i| try text.print(alloc, "{s}{s}", .{ if (i > 0) ", " else "", engine });
    try text.print(alloc, "; {s} may be partial]", .{what});
}

fn appendFailedCount(alloc: Allocator, summary: *std.ArrayList(u8), failed: usize) Allocator.Error!void {
    if (failed > 0) try summary.print(alloc, " · {d} engine{s} failed", .{ failed, if (failed == 1) "" else "s" });
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
    else if (std.mem.eql(u8, f.code, "budget"))
        std.fmt.comptimePrint("the web search budget for this turn ({d} searches) is spent; answer from what you have, or web_fetch a page already found", .{client.max_billed_calls})
    else if (std.mem.eql(u8, f.code, "not_available"))
        "web_context is unavailable on the search service now; use web_search"
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
        var r = (try run(&service, a, case.name, case.arguments, false, null)).?;
        defer r.deinit(a);
        try std.testing.expect(r.is_error);
        if (std.mem.indexOf(u8, r.text, case.needle) == null) {
            std.debug.print("expected \"{s}\" in \"{s}\"\n", .{ case.needle, r.text });
            return error.TestUnexpectedResult;
        }
    }
    try std.testing.expect((try run(&service, a, "bash", "{}", false, null)) == null);
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

test "a corrected query leads the search, and the top three results show two extra snippets each" {
    const a = std.testing.allocator;
    const extras = [_][]const u8{ "first extra", "second extra", "third extra" };
    var results: [4]client.Search.Result = undefined;
    for (&results, 0..) |*r, i| r.* = .{ .title = "T", .url = "https://e.org/", .snippet = "s", .extra_snippets = if (i == 1) &.{} else &extras };
    results[0].url = "https://e.org/1";
    var r = try renderSearch(a, .{ .query = "zgi", .altered_query = "zig", .results = &results });
    defer r.deinit(a);
    try std.testing.expectEqualStrings(
        \\searched for "zig" (the service corrected the query)
        \\1. T — https://e.org/1
        \\   s
        \\   first extra
        \\   second extra
        \\2. T — https://e.org/
        \\   s
        \\3. T — https://e.org/
        \\   s
        \\   first extra
        \\   second extra
        \\4. T — https://e.org/
        \\   s
    , r.text);
    try std.testing.expectEqualStrings("4 results", r.summary.?);
}

test "context renders sources with their passages, zero sources plainly, and cuts past 12,000 characters" {
    const a = std.testing.allocator;
    const sources = [_]client.Context.Source{
        .{ .url = "https://ziglang.org/n", .title = "Release Notes", .published = "2025-08-19", .snippets = &.{ "usingnamespace is removed.", "Before:\nconst a = 1;" } },
        .{ .url = "https://e.org/z" },
    };
    var r = try renderContext(a, .{ .query = "q", .sources = &sources, .engines_failed = &.{"searxng"} });
    defer r.deinit(a);
    try std.testing.expectEqualStrings(
        \\1. Release Notes — https://ziglang.org/n (2025-08-19)
        \\   usingnamespace is removed.
        \\   Before:
        \\   const a = 1;
        \\
        \\2. https://e.org/z — https://e.org/z
        \\[engines failed: searxng; passages may be partial]
    , r.text);
    try std.testing.expectEqualStrings("2 sources · 1 engine failed", r.summary.?);

    var none = try renderContext(a, .{ .query = "q" });
    defer none.deinit(a);
    try std.testing.expect(!none.is_error);
    try std.testing.expectEqualStrings("no passages matched the query", none.text);
    try std.testing.expectEqualStrings("no passages", none.summary.?);

    // Five sources of about 3,950 characters: three fit, two are not shown.
    const long: [3900]u8 = @splat('x');
    const snippets = [_][]const u8{&long};
    var many: [5]client.Context.Source = @splat(.{ .url = "https://e.org/", .title = "L", .snippets = &snippets });
    var cut = try renderContext(a, .{ .query = "q", .sources = &many });
    defer cut.deinit(a);
    try std.testing.expect(cut.truncated);
    try std.testing.expect(std.mem.endsWith(u8, cut.text, "\n[2 more sources not shown]"));
    try std.testing.expectEqual(@as(usize, 3), std.mem.count(u8, cut.text, "https://e.org/\n"));
    try std.testing.expect(cut.text.len < context_chars + 100);
    try std.testing.expectEqualStrings("5 sources", cut.summary.?);

    // One source alone past the bound keeps its first 12,000 characters.
    const huge: [20_000]u8 = @splat('y');
    const one = [_]client.Context.Source{.{ .url = "https://e.org/h", .snippets = &.{&huge} }};
    var alone = try renderContext(a, .{ .query = "q", .sources = &one });
    defer alone.deinit(a);
    try std.testing.expect(std.mem.endsWith(u8, alone.text, "[passage cut]"));
    try std.testing.expect(alone.text.len < context_chars + 100);
}

test "freshness: a known value is sent, an unknown one is refused before any request" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    const Fake = @import("fake.zig").Fake;
    var fake: Fake = try .init(io, &.{
        .{ .status = 200, .body = "{\"query\":\"zig\",\"results\":[]}" },
        .{ .status = 200, .body = "{\"query\":\"zig\",\"sources\":[]}" },
    });
    var task = try io.concurrent(Fake.serve, .{ &fake, io });
    defer fake.stop(io, &task);
    var buffer: [64]u8 = undefined;
    var service: client.Service = .init(a, io, fake.base(&buffer), "nuclis/test");
    defer service.deinit();

    var bad = (try run(&service, a, "web_search", "{\"query\":\"zig\",\"freshness\":\"decade\"}", false, null)).?;
    defer bad.deinit(a);
    try std.testing.expectEqualStrings("web_search: freshness must be day, week, month, or year", bad.text);
    var bad_context = (try run(&service, a, "web_context", "{\"query\":\"zig\",\"freshness\":\"pd\"}", false, null)).?;
    defer bad_context.deinit(a);
    try std.testing.expect(bad_context.is_error);

    var day = (try run(&service, a, "web_search", "{\"query\":\"zig\",\"freshness\":\"day\"}", false, null)).?;
    defer day.deinit(a);
    try std.testing.expectEqualStrings("/v1/search?q=zig&n=8&freshness=pd", fake.target(0));
    var year = (try run(&service, a, "web_context", "{\"query\":\"zig\",\"freshness\":\"year\"}", false, null)).?;
    defer year.deinit(a);
    try std.testing.expectEqualStrings("no passages matched the query", year.text);
    try std.testing.expectEqualStrings("/v1/context?q=zig&tokens=2048&urls=8&freshness=py", fake.target(1));
}

test "the budget, an unavailable context, and a bot challenge read as what to do next" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var spent = try failure(a, "web_search", .{ .code = "budget", .message = "" });
    defer spent.deinit(a);
    try std.testing.expectEqualStrings("web_search: the web search budget for this turn (6 searches) is spent; answer from what you have, or web_fetch a page already found", spent.text);

    const Fake = @import("fake.zig").Fake;
    const challenge = "{\"error\":{\"code\":\"upstream_failed\",\"message\":\"stackoverflow.com answered with a bot challenge (Cloudflare) that only a browser passes; try /v1/context, whose provider has the page's text\"}}";
    var fake: Fake = try .init(io, &.{
        .{ .status = 501, .body = "{\"error\":{\"code\":\"not_available\",\"message\":\"no Brave key on this host\"}}" },
        .{ .status = 404, .body = "{\"error\":{\"code\":\"not_found\",\"message\":\"no endpoint /v1/context\"}}" },
        .{ .status = 502, .body = challenge },
        .{ .status = 502, .body = challenge },
    });
    var task = try io.concurrent(Fake.serve, .{ &fake, io });
    defer fake.stop(io, &task);
    var buffer: [64]u8 = undefined;
    var service: client.Service = .init(a, io, fake.base(&buffer), "nuclis/test");
    defer service.deinit();

    for (0..2) |_| {
        var unavailable = (try run(&service, a, "web_context", "{\"query\":\"zig\"}", true, null)).?;
        defer unavailable.deinit(a);
        try std.testing.expectEqualStrings("web_context: web_context is unavailable on the search service now; use web_search", unavailable.text);
    }
    var offered = (try run(&service, a, "web_fetch", "{\"url\":\"https://stackoverflow.com/q/1\"}", true, null)).?;
    defer offered.deinit(a);
    try std.testing.expectEqualStrings("web_fetch: stackoverflow.com is behind a bot challenge only a browser passes; web_context may hold its text", offered.text);
    var alone = (try run(&service, a, "web_fetch", "{\"url\":\"https://stackoverflow.com/q/1\"}", false, null)).?;
    defer alone.deinit(a);
    try std.testing.expectEqualStrings("web_fetch: stackoverflow.com is behind a bot challenge only a browser passes", alone.text);
}
