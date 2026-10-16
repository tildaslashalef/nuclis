//! The client of the search service behind `web_search` and `web_fetch`:
//! holocron's v1 contract, whose client copy is docs/app/agent.md § 11.
//!
//! One `Service` per process holds the base URL and a pooled
//! `std.http.Client`. Each call returns a `Reply` whose arena owns everything
//! in it. Expected failures (the service's error codes, an unreachable
//! service, a timeout, a malformed answer, a cancellation) are `Failure`
//! values, never Zig errors; only allocation failure is. The request runs in
//! a concurrent task so the caller's `Beat` can repaint and cancel meanwhile.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

/// The most body bytes read from any answer; a fetch's 100,000 code points
/// of text fit with room for the JSON around them.
pub const max_body_bytes: usize = 512 * 1024;
/// Above the service's own budgets (10 s and 20 s), so its timeout error
/// arrives before ours.
pub const search_timeout_ms: u64 = 15_000;
pub const fetch_timeout_ms: u64 = 30_000;
pub const health_timeout_ms: u64 = 5_000;
const beat_ms: u64 = 50;

/// Called about every 50 ms while a request is in flight, on the caller's
/// thread; returning true cancels the request.
pub const Beat = struct {
    context: *anyopaque,
    call: *const fn (context: *anyopaque) bool,
};

pub const Failure = struct {
    /// The service's code (`busy`, `blocked_address`, …) or one of ours:
    /// `unreachable`, `timeout`, `cancelled`, `invalid_response`.
    code: []const u8,
    message: []const u8,
    /// The HTTP status of the answer; 0 when there was none.
    status: u16 = 0,
};

pub const Search = struct {
    query: []const u8 = "",
    provider: []const u8 = "",
    results: []const Result = &.{},
    engines_failed: []const []const u8 = &.{},

    pub const Result = struct {
        title: []const u8 = "",
        url: []const u8,
        snippet: []const u8 = "",
        engine: []const u8 = "",
        published: ?[]const u8 = null,
    };
};

pub const Fetched = struct {
    url: []const u8 = "",
    final_url: []const u8 = "",
    status: u16 = 0,
    content_type: []const u8 = "",
    title: []const u8 = "",
    text: []const u8 = "",
    /// Code points, as `chars`.
    start: u64 = 0,
    chars: u64 = 0,
    truncated: bool = false,
};

pub const Health = struct {
    version: []const u8 = "",
    providers: std.json.ArrayHashMap([]const u8) = .{},

    /// True when the service answered and every provider is `ok`.
    pub fn ok(self: Health) bool {
        if (self.providers.map.count() == 0) return false;
        for (self.providers.map.values()) |state| if (!std.mem.eql(u8, state, "ok")) return false;
        return true;
    }
};

pub fn Reply(comptime T: type) type {
    return struct {
        arena: std.heap.ArenaAllocator,
        outcome: union(enum) { ok: T, failed: Failure },

        pub fn deinit(self: *@This()) void {
            self.arena.deinit();
            self.* = undefined;
        }
    };
}

/// `https`, or `http` on a loopback host (a local stand-in such as
/// `scripts/search-gateway-stub.py`); a host, and no credentials, query, or
/// fragment. The configuration and the client apply the same rule.
pub fn validBase(url: []const u8) bool {
    const uri = std.Uri.parse(url) catch return false;
    const host = uri.host orelse return false;
    if (uri.user != null or uri.password != null or uri.query != null or uri.fragment != null) return false;
    if (std.ascii.eqlIgnoreCase(uri.scheme, "https")) return true;
    if (!std.ascii.eqlIgnoreCase(uri.scheme, "http")) return false;
    var buffer: [std.Io.net.HostName.max_len]u8 = undefined;
    const name = host.toRaw(&buffer) catch return false;
    return std.ascii.eqlIgnoreCase(name, "localhost") or std.mem.eql(u8, name, "127.0.0.1") or std.mem.eql(u8, name, "::1") or std.mem.eql(u8, name, "[::1]");
}

pub const Service = struct {
    http: std.http.Client,
    /// The base URL without its trailing slash; borrowed, it must outlive
    /// the service.
    base: []const u8,
    /// `nuclis/<version>`; borrowed.
    user_agent: []const u8,

    /// `gpa` must be thread safe (`std.http.Client`'s pool is shared with
    /// the request task). `base` must pass `validBase`.
    pub fn init(gpa: Allocator, io: Io, base: []const u8, user_agent: []const u8) Service {
        std.debug.assert(validBase(base));
        return .{ .http = .{ .allocator = gpa, .io = io }, .base = std.mem.trimEnd(u8, base, "/"), .user_agent = user_agent };
    }

    pub fn deinit(self: *Service) void {
        self.http.deinit();
        self.* = undefined;
    }

    /// `GET /v1/search?q=&n=`.
    pub fn search(self: *Service, gpa: Allocator, query: []const u8, count: u8, beat: ?Beat) Allocator.Error!Reply(Search) {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        var url: std.ArrayList(u8) = .empty;
        try url.print(a, "{s}/v1/search?q=", .{self.base});
        try appendEscaped(a, &url, query);
        try url.print(a, "&n={d}", .{count});
        const outcome = try self.call(Search, a, url.items, search_timeout_ms, beat);
        return .{ .arena = arena, .outcome = switch (outcome) {
            .ok => |v| .{ .ok = v },
            .failed => |f| .{ .failed = f },
        } };
    }

    /// `GET /v1/fetch?url=&start=&max_chars=`.
    pub fn fetch(self: *Service, gpa: Allocator, target: []const u8, start: u64, max_chars: u32, beat: ?Beat) Allocator.Error!Reply(Fetched) {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        var url: std.ArrayList(u8) = .empty;
        try url.print(a, "{s}/v1/fetch?url=", .{self.base});
        try appendEscaped(a, &url, target);
        try url.print(a, "&start={d}&max_chars={d}", .{ start, max_chars });
        const outcome = try self.call(Fetched, a, url.items, fetch_timeout_ms, beat);
        return .{ .arena = arena, .outcome = switch (outcome) {
            .ok => |v| .{ .ok = v },
            .failed => |f| .{ .failed = f },
        } };
    }

    /// `GET /v1/health`.
    pub fn health(self: *Service, gpa: Allocator, beat: ?Beat) Allocator.Error!Reply(Health) {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const url = try std.fmt.allocPrint(a, "{s}/v1/health", .{self.base});
        const outcome = try self.call(Health, a, url, health_timeout_ms, beat);
        return .{ .arena = arena, .outcome = switch (outcome) {
            .ok => |v| .{ .ok = v },
            .failed => |f| .{ .failed = f },
        } };
    }

    fn Outcome(comptime T: type) type {
        return union(enum) { ok: T, failed: Failure };
    }

    /// One request and its decoding; everything lands in `a`.
    fn call(self: *Service, comptime T: type, a: Allocator, url: []const u8, timeout_ms: u64, beat: ?Beat) Allocator.Error!Outcome(T) {
        const raw = switch (try self.exchange(a, url, timeout_ms, beat)) {
            .done => |raw| raw,
            .failed => |f| return .{ .failed = f },
        };
        if (raw.status == 200) {
            const value = std.json.parseFromSliceLeaky(T, a, raw.body, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return .{ .failed = .{ .code = "invalid_response", .message = try std.fmt.allocPrint(a, "the search service's answer is not the v1 shape ({s})", .{@errorName(err)}), .status = raw.status } },
            };
            return .{ .ok = value };
        }
        const Body = struct { @"error": struct { code: []const u8, message: []const u8 = "" } };
        const parsed = std.json.parseFromSliceLeaky(Body, a, raw.body, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            // A proxy in front of the service (the edge) answers in HTML.
            else => return .{ .failed = .{ .code = "invalid_response", .message = try std.fmt.allocPrint(a, "the search service answered {d} without an error body", .{raw.status}), .status = raw.status } },
        };
        return .{ .failed = .{ .code = parsed.@"error".code, .message = parsed.@"error".message, .status = raw.status } };
    }

    const Raw = struct { status: u16, body: []const u8 };
    const Exchanged = union(enum) { done: Raw, failed: Failure };

    /// Runs the request in a task and waits for it in beats, so the deadline
    /// and the caller's cancellation are honoured while it blocks. The task is
    /// joined before returning: nothing it borrows outlives this call.
    fn exchange(self: *Service, a: Allocator, url: []const u8, timeout_ms: u64, beat: ?Beat) Allocator.Error!Exchanged {
        const io = self.http.io;
        const Completion = union(enum) { request: anyerror!void, beat: Io.Cancelable!void };
        var buffer: [2]Completion = undefined;
        var select = Io.Select(Completion).init(io, &buffer);
        defer select.cancelDiscard();
        var raw: ?Raw = null;
        select.concurrent(.request, perform, .{ self, a, url, &raw }) catch
            return .{ .failed = .{ .code = "unreachable", .message = "the search service cannot be called: this process runs no concurrent tasks" } };
        const started = Io.Clock.awake.now(io);
        while (true) {
            select.concurrent(.beat, sleep, .{ io, beat_ms }) catch
                return .{ .failed = .{ .code = "unreachable", .message = "the search service cannot be called: this process runs no concurrent tasks" } };
            const done = select.await() catch return .{ .failed = .{ .code = "cancelled", .message = "the request was cancelled" } };
            switch (done) {
                .request => |result| {
                    result catch |err| return .{ .failed = try transportFailure(a, self.base, err) };
                    return .{ .done = raw orelse return .{ .failed = .{ .code = "invalid_response", .message = "the search service sent no answer" } } };
                },
                .beat => {
                    if (beat) |b| if (b.call(b.context)) return .{ .failed = .{ .code = "cancelled", .message = "the request was cancelled" } };
                    const elapsed: u64 = @intCast(started.durationTo(Io.Clock.awake.now(io)).nanoseconds);
                    if (elapsed >= timeout_ms * std.time.ns_per_ms)
                        return .{ .failed = .{ .code = "timeout", .message = try std.fmt.allocPrint(a, "the search service did not answer within {d} s", .{timeout_ms / 1000}) } };
                },
            }
        }
    }

    fn perform(self: *Service, a: Allocator, url: []const u8, out: *?Raw) !void {
        const uri = try std.Uri.parse(url);
        var req = try self.http.request(.GET, uri, .{
            .redirect_behavior = .unhandled,
            .headers = .{ .user_agent = .{ .override = self.user_agent }, .accept_encoding = .{ .override = "identity" } },
            .extra_headers = &.{.{ .name = "Accept", .value = "application/json" }},
        });
        defer req.deinit();
        try req.sendBodiless();
        var res = try req.receiveHead(&.{});
        const status: u16 = @backingInt(res.head.status);
        if (res.head.content_encoding != .identity) return error.UnsupportedContentEncoding;
        if (res.head.content_length) |n| if (n > max_body_bytes) return error.ResponseTooLarge;
        var transfer: [16 * 1024]u8 = undefined;
        const reader = res.reader(&transfer);
        var body: std.ArrayList(u8) = .empty;
        reader.appendRemaining(a, &body, .limited(max_body_bytes + 1)) catch |err| switch (err) {
            error.StreamTooLong => return error.ResponseTooLarge,
            else => return err,
        };
        if (body.items.len > max_body_bytes) return error.ResponseTooLarge;
        out.* = .{ .status = status, .body = body.items };
    }

    fn sleep(io: Io, ms: u64) Io.Cancelable!void {
        try io.sleep(.fromMilliseconds(@intCast(ms)), .awake);
    }
};

/// What the model and the transcript read for a request that never got an
/// answer: one sentence naming the cause.
fn transportFailure(a: Allocator, base: []const u8, err: anyerror) Allocator.Error!Failure {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.ResponseTooLarge => .{ .code = "invalid_response", .message = try std.fmt.allocPrint(a, "the search service's answer exceeded {d} KiB", .{max_body_bytes / 1024}) },
        error.Canceled => .{ .code = "cancelled", .message = "the request was cancelled" },
        else => .{ .code = "unreachable", .message = try std.fmt.allocPrint(a, "the search service at {s} could not be reached ({s})", .{ base, @errorName(err) }) },
    };
}

/// Percent-encodes everything but RFC 3986's unreserved characters.
fn appendEscaped(a: Allocator, out: *std.ArrayList(u8), value: []const u8) Allocator.Error!void {
    for (value) |c| switch (c) {
        'A'...'Z', 'a'...'z', '0'...'9', '-', '.', '_', '~' => try out.append(a, c),
        else => try out.print(a, "%{X:0>2}", .{c}),
    };
}

test "the base URL rule: https anywhere, http on loopback only" {
    for ([_][]const u8{ "https://search.apps.lab.nuclis.dev", "https://search.example/", "https://search.example/prefix", "http://127.0.0.1:8099", "http://localhost:8099/", "http://[::1]:8099" }) |url|
        try std.testing.expect(validBase(url));
    for ([_][]const u8{ "http://search.example", "ftp://search.example", "https://me:pw@search.example", "https://search.example/?key=1", "https://search.example/#x", "search.example", "" }) |url|
        try std.testing.expect(!validBase(url));
}

test "query values are percent-encoded except the unreserved set" {
    const a = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    try appendEscaped(a, &out, "zig std.http?a=b&c d/é~");
    try std.testing.expectEqualStrings("zig%20std.http%3Fa%3Db%26c%20d%2F%C3%A9~", out.items);
}

test "health is ok only when every provider is" {
    var health: Health = .{};
    try std.testing.expect(!health.ok());
    defer health.providers.deinit(std.testing.allocator);
    try health.providers.map.put(std.testing.allocator, "searxng", "ok");
    try std.testing.expect(health.ok());
    try health.providers.map.put(std.testing.allocator, "brave", "down");
    try std.testing.expect(!health.ok());
}

const Fake = @import("fake.zig").Fake;

test "search and fetch decode the v1 answers, ignoring fields they do not know" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var fake: Fake = try .init(io, &.{
        .{ .status = 200, .body =
        \\{"query":"zig http","provider":"searxng","new_field":1,"results":[
        \\{"title":"Zig","url":"https://ziglang.org/","snippet":"A language","engine":"duckduckgo","published":"2026-09-01"},
        \\{"title":"zig","url":"https://github.com/ziglang/zig","snippet":"","engine":"github","published":null}],
        \\"engines_failed":["mojeek"]}
        },
        .{ .status = 200, .body =
        \\{"url":"https://example.org/a","final_url":"https://example.org/a/","status":200,"content_type":"text/html",
        \\"title":"A","text":"# A\n\nbody","start":0,"chars":48211,"truncated":true}
        },
    });
    var task = try io.concurrent(Fake.serve, .{ &fake, io });
    defer fake.stop(io, &task);
    var buffer: [64]u8 = undefined;
    var service: Service = .init(a, io, fake.base(&buffer), "nuclis/test");
    defer service.deinit();

    var found = try service.search(a, "zig http&more", 8, null);
    defer found.deinit();
    const s = found.outcome.ok;
    try std.testing.expectEqualStrings("searxng", s.provider);
    try std.testing.expectEqual(@as(usize, 2), s.results.len);
    try std.testing.expectEqualStrings("2026-09-01", s.results[0].published.?);
    try std.testing.expect(s.results[1].published == null);
    try std.testing.expectEqualStrings("mojeek", s.engines_failed[0]);
    try std.testing.expectEqualStrings("/v1/search?q=zig%20http%26more&n=8", fake.target(0));
    try std.testing.expectEqualStrings("nuclis/test", fake.agent[0..fake.agent_len]);

    var page = try service.fetch(a, "https://example.org/a?b=1", 100, 2000, null);
    defer page.deinit();
    const f = page.outcome.ok;
    try std.testing.expectEqual(@as(u64, 48211), f.chars);
    try std.testing.expect(f.truncated);
    try std.testing.expectEqualStrings("# A\n\nbody", f.text);
    try std.testing.expectEqualStrings("/v1/fetch?url=https%3A%2F%2Fexample.org%2Fa%3Fb%3D1&start=100&max_chars=2000", fake.target(1));
}

test "the service's error codes, a proxy's page, and a malformed answer are failures" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var fake: Fake = try .init(io, &.{
        .{ .status = 503, .body = "{\"error\":{\"code\":\"busy\",\"message\":\"every slot stayed taken for 2 s\"}}" },
        .{ .status = 502, .body = "<html>Bad Gateway</html>" },
        .{ .status = 200, .body = "{\"results\": 7}" },
        .{ .status = 200, .body = "{\"version\":\"0.1.0\",\"providers\":{\"searxng\":\"down\"}}" },
    });
    var task = try io.concurrent(Fake.serve, .{ &fake, io });
    defer fake.stop(io, &task);
    var buffer: [64]u8 = undefined;
    var service: Service = .init(a, io, fake.base(&buffer), "nuclis/test");
    defer service.deinit();

    var busy = try service.search(a, "q", 8, null);
    defer busy.deinit();
    try std.testing.expectEqualStrings("busy", busy.outcome.failed.code);
    try std.testing.expectEqual(@as(u16, 503), busy.outcome.failed.status);
    var proxy = try service.search(a, "q", 8, null);
    defer proxy.deinit();
    try std.testing.expectEqualStrings("invalid_response", proxy.outcome.failed.code);
    try std.testing.expect(std.mem.indexOf(u8, proxy.outcome.failed.message, "502") != null);
    var malformed = try service.search(a, "q", 8, null);
    defer malformed.deinit();
    try std.testing.expectEqualStrings("invalid_response", malformed.outcome.failed.code);
    var health = try service.health(a, null);
    defer health.deinit();
    try std.testing.expect(!health.outcome.ok.ok());
}

test "an unreachable service, a late answer, and a cancelling beat are failures" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var fake: Fake = try .init(io, &.{
        .{ .status = 200, .body = "{}", .delay_ms = 400 },
        .{ .status = 200, .body = "{}", .delay_ms = 400 },
    });
    var task = try io.concurrent(Fake.serve, .{ &fake, io });
    var stopped = false;
    defer if (!stopped) fake.stop(io, &task);
    var buffer: [64]u8 = undefined;
    const base_url = fake.base(&buffer);
    var service: Service = .init(a, io, base_url, "nuclis/test");
    defer service.deinit();

    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const url = try std.fmt.allocPrint(arena.allocator(), "{s}/v1/health", .{service.base});
    const late = try service.call(Health, arena.allocator(), url, 100, null);
    try std.testing.expectEqualStrings("timeout", late.failed.code);

    const Cancel = struct {
        fn now(_: *anyopaque) bool {
            return true;
        }
    };
    var dummy: u8 = 0;
    var cancelled = try service.health(a, .{ .context = &dummy, .call = Cancel.now });
    defer cancelled.deinit();
    try std.testing.expectEqualStrings("cancelled", cancelled.outcome.failed.code);

    fake.stop(io, &task);
    stopped = true;
    var gone = try service.health(a, null);
    defer gone.deinit();
    try std.testing.expectEqualStrings("unreachable", gone.outcome.failed.code);
}
