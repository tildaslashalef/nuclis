//! The client of the search service behind `web_search`, `web_fetch`, and
//! `web_context`: the v1 contract in docs/app/agent.md § 11.
//!
//! One `Service` per process holds the base URL and a pooled
//! `std.http.Client`. Each call returns a `Reply` whose arena owns everything
//! in it. Expected failures (the service's error codes, an unreachable
//! service, a timeout, a malformed answer, a cancellation) are `Failure`
//! values, never Zig errors; only allocation failure is. The request runs in
//! a concurrent task so the caller's `Beat` can repaint and cancel meanwhile.
//! Searches, contexts, and fetches that ask a question are billed upstream,
//! so a `Service` answers at most `max_billed_calls` uncached ones between
//! `beginTurn`s. A failure is judged by its code, never its message (`Retry`).
const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

/// The most body bytes read from any answer; a fetch's 100,000 code points
/// of text fit with room for the JSON around them.
pub const max_body_bytes: usize = 512 * 1024;
/// Above the service's own budgets (10 s search, 20 s context, 25 s fetch),
/// so its timeout error arrives before ours.
pub const search_timeout_ms: u64 = 15_000;
pub const fetch_timeout_ms: u64 = 30_000;
pub const context_timeout_ms: u64 = 25_000;
pub const health_timeout_ms: u64 = 5_000;
/// The probe at startup: a slow service starts without `web_context`.
pub const probe_timeout_ms: u64 = 2_000;
/// What one `web_context` call asks for: about 2K tokens from at most 8 pages.
pub const context_tokens: u16 = 2048;
pub const context_urls: u8 = 8;
/// Uncached searches, contexts, and fetches with a question one turn (an
/// agent turn, an API request) may make: each is billed upstream.
pub const max_billed_calls: u8 = 6;
/// The longest question a fetch may ask, as the service allows.
pub const max_question_bytes = 400;
const beat_ms: u64 = 50;

/// Called about every 50 ms while a request is in flight, on the caller's
/// thread; returning true cancels the request.
pub const Beat = struct {
    context: *anyopaque,
    call: *const fn (context: *anyopaque) bool,
};

pub const Failure = struct {
    /// The service's code (`busy`, `blocked_address`, …; `Retry.classOf`
    /// sorts them) or one of ours:
    /// `unreachable`, `timeout`, `cancelled`, `invalid_response`, `budget`.
    code: []const u8,
    message: []const u8,
    /// The HTTP status of the answer; 0 when there was none.
    status: u16 = 0,
};

pub const Search = struct {
    query: []const u8 = "",
    /// The corrected query the provider searched instead, if it did.
    altered_query: ?[]const u8 = null,
    /// The provider that answered; a whole answer, never a partial one.
    provider: []const u8 = "",
    results: []const Result = &.{},
    /// The providers skipped before `provider` answered.
    providers_failed: []const []const u8 = &.{},

    pub const Result = struct {
        title: []const u8 = "",
        url: []const u8,
        snippet: []const u8 = "",
        /// Up to five more passages from the page.
        snippets: []const []const u8 = &.{},
        published: ?[]const u8 = null,
    };
};

/// `/v1/context`: passages the provider picked from several pages.
pub const Context = struct {
    query: []const u8 = "",
    provider: []const u8 = "",
    sources: []const Source = &.{},
    providers_failed: []const []const u8 = &.{},

    pub const Source = struct {
        url: []const u8,
        title: []const u8 = "",
        published: ?[]const u8 = null,
        snippets: []const []const u8 = &.{},
    };
};

/// How recent a search's or a context's pages must be. The contract's date
/// ranges are not offered: the model would have to know today's date.
pub const Freshness = enum {
    day,
    week,
    month,
    year,

    pub fn param(self: Freshness) []const u8 {
        return switch (self) {
            .day => "pd",
            .week => "pw",
            .month => "pm",
            .year => "py",
        };
    }
};

pub const Fetched = struct {
    url: []const u8 = "",
    final_url: []const u8 = "",
    provider: []const u8 = "",
    providers_failed: []const []const u8 = &.{},
    /// Null when the provider does not report them (Parallel).
    status: ?u16 = null,
    content_type: ?[]const u8 = null,
    title: []const u8 = "",
    published: ?[]const u8 = null,
    /// The text is only the parts that answer the fetch's question.
    focused: bool = false,
    text: []const u8 = "",
    /// Code points, as `chars`.
    start: u64 = 0,
    chars: u64 = 0,
    truncated: bool = false,
};

pub const Health = struct {
    version: []const u8 = "",
    providers: std.json.ArrayHashMap([]const u8) = .{},
    /// The paths this service answers beyond the first three (`/v1/context`).
    endpoints: []const []const u8 = &.{},

    /// True when any provider is `ok`: each endpoint tries its providers in
    /// order, so one exhausted provider still leaves the endpoint working.
    pub fn ok(self: Health) bool {
        for (self.providers.map.values()) |state| if (std.mem.eql(u8, state, "ok")) return true;
        return false;
    }

    pub fn has(self: Health, endpoint: []const u8) bool {
        for (self.endpoints) |e| if (std.mem.eql(u8, e, endpoint)) return true;
        return false;
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
/// `scripts/app/search-gateway-stub.py`); a host, and no credentials, query, or
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
    /// Uncached searches and contexts since `beginTurn`. One turn at a time
    /// uses a service (the agent's loop, serve's GPU worker), so no lock.
    billed: u8 = 0,
    /// The service answers `/v1/context`, as `probe` found it.
    context_offered: bool = false,
    /// The wait before a transient failure is sent again; tests shorten it.
    retry_pause_ms: u64 = 1_000,

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

    /// Starts a turn: the allowance of uncached calls is whole again.
    pub fn beginTurn(self: *Service) void {
        self.billed = 0;
    }

    /// `GET /v1/search?q=&n=[&freshness=]`; `budget` once the turn's
    /// allowance is spent, without a request.
    pub fn search(self: *Service, gpa: Allocator, query: []const u8, count: u8, freshness: ?Freshness, beat: ?Beat) Allocator.Error!Reply(Search) {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        if (self.spent()) return .{ .arena = arena, .outcome = .{ .failed = spentFailure() } };
        var url: std.ArrayList(u8) = .empty;
        try url.print(a, "{s}/v1/search?q=", .{self.base});
        try appendEscaped(a, &url, query);
        try url.print(a, "&n={d}", .{count});
        if (freshness) |f| try url.print(a, "&freshness={s}", .{f.param()});
        const outcome = try self.call(Search, a, url.items, search_timeout_ms, true, beat);
        return .{ .arena = arena, .outcome = switch (outcome) {
            .ok => |v| .{ .ok = v },
            .failed => |f| .{ .failed = f },
        } };
    }

    /// `GET /v1/context?q=&tokens=&urls=[&freshness=]`, at the host's fixed
    /// budget; `budget` once the turn's allowance is spent, as `search`.
    pub fn context(self: *Service, gpa: Allocator, query: []const u8, freshness: ?Freshness, beat: ?Beat) Allocator.Error!Reply(Context) {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        if (self.spent()) return .{ .arena = arena, .outcome = .{ .failed = spentFailure() } };
        var url: std.ArrayList(u8) = .empty;
        try url.print(a, "{s}/v1/context?q=", .{self.base});
        try appendEscaped(a, &url, query);
        try url.print(a, "&tokens={d}&urls={d}", .{ context_tokens, context_urls });
        if (freshness) |f| try url.print(a, "&freshness={s}", .{f.param()});
        const outcome = try self.call(Context, a, url.items, context_timeout_ms, true, beat);
        return .{ .arena = arena, .outcome = switch (outcome) {
            .ok => |v| .{ .ok = v },
            .failed => |f| .{ .failed = f },
        } };
    }

    fn spent(self: *const Service) bool {
        return self.billed >= max_billed_calls;
    }

    fn spentFailure() Failure {
        return .{ .code = "budget", .message = std.fmt.comptimePrint("the turn's {d} uncached billed calls are spent", .{max_billed_calls}) };
    }

    /// `GET /v1/fetch?url=&start=&max_chars=[&q=]`. A `question` (at most
    /// `max_question_bytes`) asks for only the parts that answer it; that is
    /// billed, so it counts against the allowance as `search` does, while a
    /// plain fetch is free.
    pub fn fetch(self: *Service, gpa: Allocator, target: []const u8, start: u64, max_chars: u32, question: ?[]const u8, beat: ?Beat) Allocator.Error!Reply(Fetched) {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        if (question != null and self.spent()) return .{ .arena = arena, .outcome = .{ .failed = spentFailure() } };
        var url: std.ArrayList(u8) = .empty;
        try url.print(a, "{s}/v1/fetch?url=", .{self.base});
        try appendEscaped(a, &url, target);
        try url.print(a, "&start={d}&max_chars={d}", .{ start, max_chars });
        if (question) |q| {
            try url.appendSlice(a, "&q=");
            try appendEscaped(a, &url, q);
        }
        const outcome = try self.call(Fetched, a, url.items, fetch_timeout_ms, question != null, beat);
        return .{ .arena = arena, .outcome = switch (outcome) {
            .ok => |v| .{ .ok = v },
            .failed => |f| .{ .failed = f },
        } };
    }

    /// `GET /v1/health`.
    pub fn health(self: *Service, gpa: Allocator, beat: ?Beat) Allocator.Error!Reply(Health) {
        return self.healthWithin(gpa, health_timeout_ms, beat);
    }

    /// Asks the service once what it offers and sets `context_offered`; any
    /// failure leaves it false.
    pub fn probe(self: *Service, gpa: Allocator) Allocator.Error!void {
        var reply = try self.healthWithin(gpa, probe_timeout_ms, null);
        defer reply.deinit();
        self.context_offered = switch (reply.outcome) {
            .ok => |h| h.has("/v1/context"),
            .failed => false,
        };
    }

    fn healthWithin(self: *Service, gpa: Allocator, timeout_ms: u64, beat: ?Beat) Allocator.Error!Reply(Health) {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const url = try std.fmt.allocPrint(a, "{s}/v1/health", .{self.base});
        const outcome = try self.call(Health, a, url, timeout_ms, false, beat);
        return .{ .arena = arena, .outcome = switch (outcome) {
            .ok => |v| .{ .ok = v },
            .failed => |f| .{ .failed = f },
        } };
    }

    fn Outcome(comptime T: type) type {
        return union(enum) { ok: T, failed: Failure };
    }

    /// One request and its decoding; everything lands in `a`. An answer
    /// whose code `Retry` calls transient is sent once more after
    /// `retry_pause_ms`, each attempt under `timeout_ms`. A `metered` call
    /// counts once against the allowance unless the service says its answer
    /// came from its cache.
    fn call(self: *Service, comptime T: type, a: Allocator, url: []const u8, timeout_ms: u64, metered: bool, beat: ?Beat) Allocator.Error!Outcome(T) {
        var raw = switch (try self.exchange(a, url, timeout_ms, beat)) {
            .done => |raw| raw,
            .failed => |f| return .{ .failed = f },
        };
        if (raw.status != 200 and Retry.classOf((try failureOf(a, raw)).code) == .transient) {
            if (self.pause(beat)) return .{ .failed = .{ .code = "cancelled", .message = "the request was cancelled" } };
            raw = switch (try self.exchange(a, url, timeout_ms, beat)) {
                .done => |again| again,
                .failed => |f| return .{ .failed = f },
            };
        }
        if (metered and !raw.cached) self.billed +|= 1;
        if (raw.status != 200) return .{ .failed = try failureOf(a, raw) };
        const value = std.json.parseFromSliceLeaky(T, a, raw.body, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return .{ .failed = .{ .code = "invalid_response", .message = try std.fmt.allocPrint(a, "the search service's answer is not the v1 shape ({s})", .{@errorName(err)}), .status = raw.status } },
        };
        return .{ .ok = value };
    }

    /// Waits `retry_pause_ms` in beats; true when the beat cancelled it.
    fn pause(self: *Service, beat: ?Beat) bool {
        var waited: u64 = 0;
        while (waited < self.retry_pause_ms) : (waited += beat_ms) {
            if (beat) |b| if (b.call(b.context)) return true;
            sleep(self.http.io, @min(beat_ms, self.retry_pause_ms - waited)) catch return true;
        }
        return false;
    }

    /// A non-200 answer as the failure it names.
    fn failureOf(a: Allocator, raw: Raw) Allocator.Error!Failure {
        const Body = struct { @"error": struct { code: []const u8, message: []const u8 = "" } };
        const parsed = std.json.parseFromSliceLeaky(Body, a, raw.body, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            // A proxy in front of the service (the edge) answers in HTML.
            else => return .{ .code = "invalid_response", .message = try std.fmt.allocPrint(a, "the search service answered {d} without an error body", .{raw.status}), .status = raw.status },
        };
        return .{ .code = parsed.@"error".code, .message = parsed.@"error".message, .status = raw.status };
    }

    /// `cached`: `X-Search-Cache` said `hit` or `stale`; a missing header
    /// (the stub, an older service) counts as a miss.
    const Raw = struct { status: u16, body: []const u8, cached: bool = false };
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

    /// One request, sent once more on a fresh connection when `Retry` says
    /// the pooled connection it reused was dead before the answer began.
    /// Both attempts run inside one `exchange`, under its deadline and beat.
    fn perform(self: *Service, a: Allocator, url: []const u8, out: *?Raw) !void {
        const uri = try std.Uri.parse(url);
        const pooled = try self.idleConnections() > 0;
        var answered = false;
        self.attempt(a, uri, out, &answered) catch |err| {
            if (!pooled or answered or !Retry.afterTransport(err)) return err;
            try self.closeIdle();
            answered = false;
            return self.attempt(a, uri, out, &answered);
        };
    }

    /// `answered` turns true once the answer's head arrived. A failure to
    /// send or to read the head returns the socket's own error
    /// (`ConnectionResetByPeer`, `BrokenPipe`) rather than the stream's
    /// `WriteFailed` or `ReadFailed`, so `Retry` can tell them apart.
    fn attempt(self: *Service, a: Allocator, uri: std.Uri, out: *?Raw, answered: *bool) !void {
        var req = try self.http.request(.GET, uri, .{
            .redirect_behavior = .unhandled,
            .headers = .{ .user_agent = .{ .override = self.user_agent }, .accept_encoding = .{ .override = "identity" } },
            .extra_headers = &.{.{ .name = "Accept", .value = "application/json" }},
        });
        defer req.deinit();
        req.sendBodiless() catch |err| switch (err) {
            error.WriteFailed => return req.connection.?.stream_writer.err orelse err,
        };
        var res = req.receiveHead(&.{}) catch |err| switch (err) {
            error.ReadFailed => return req.connection.?.getReadError() orelse err,
            else => return err,
        };
        answered.* = true;
        const status: u16 = @backingInt(res.head.status);
        // Read before the body: `reader` invalidates the head's bytes.
        var cached = false;
        var headers = res.head.iterateHeaders();
        while (headers.next()) |header| if (std.ascii.eqlIgnoreCase(header.name, "x-search-cache")) {
            cached = std.ascii.eqlIgnoreCase(header.value, "hit") or std.ascii.eqlIgnoreCase(header.value, "stale");
        };
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
        out.* = .{ .status = status, .body = body.items, .cached = cached };
    }

    /// Idle connections in the pool, which the next request reuses: every
    /// connection this client opens is to `base`.
    fn idleConnections(self: *Service) Io.Cancelable!usize {
        const pool = &self.http.connection_pool;
        try pool.mutex.lock(self.http.io);
        defer pool.mutex.unlock(self.http.io);
        return pool.free_len;
    }

    /// Closes every idle pooled connection, so the next request opens its own.
    fn closeIdle(self: *Service) Io.Cancelable!void {
        const pool = &self.http.connection_pool;
        const size = pool.free_size;
        try pool.resize(self.http.io, 0);
        try pool.resize(self.http.io, size);
    }

    fn sleep(io: Io, ms: u64) Io.Cancelable!void {
        try io.sleep(.fromMilliseconds(@intCast(ms)), .awake);
    }
};

/// What a failure means, by its cause: the service's error code (never its
/// message) or, for a request that got no answer, the transport error. Every
/// request is a `GET`, so a replay is safe; each cause is retried at most
/// once per call, and the call counts once against the allowance.
pub const Retry = struct {
    pub const Class = enum {
        /// May work a moment later: sent once more, then reported.
        transient,
        /// Will not change: reported at once.
        permanent,
        /// The page's own server failed (a 404, a 403): reported, and another
        /// page may serve.
        page,
    };

    const table = [_]struct { []const u8, Class }{
        .{ "busy", .transient },
        .{ "upstream_timeout", .transient },
        .{ "provider_unavailable", .transient },
        .{ "blocked_address", .permanent },
        .{ "bad_request", .permanent },
        .{ "unsupported_content", .permanent },
        .{ "not_available", .permanent },
        .{ "upstream_failed", .page },
    };

    /// A code the table lacks (one of ours, or one a newer service added) is
    /// permanent: it is never sent again.
    pub fn classOf(code: []const u8) Class {
        for (table) |entry| if (std.mem.eql(u8, entry[0], code)) return entry[1];
        return .permanent;
    }

    /// A pooled connection the peer closed while it sat idle (the edge in
    /// front of the service drops idle ones after about 5 minutes): the
    /// request failed before any byte of the answer arrived.
    fn afterTransport(err: anyerror) bool {
        return switch (err) {
            error.HttpConnectionClosing, error.ConnectionResetByPeer, error.EndOfStream, error.BrokenPipe => true,
            else => false,
        };
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
    for ([_][]const u8{ "https://search.home.example", "https://search.example/", "https://search.example/prefix", "http://127.0.0.1:8099", "http://localhost:8099/", "http://[::1]:8099" }) |url|
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

test "health is ok when any provider is" {
    var health: Health = .{};
    try std.testing.expect(!health.ok());
    defer health.providers.deinit(std.testing.allocator);
    try health.providers.map.put(std.testing.allocator, "parallel", "down");
    try std.testing.expect(!health.ok());
    try health.providers.map.put(std.testing.allocator, "brave", "ok");
    try std.testing.expect(health.ok());
}

test "freshness maps to the contract's codes" {
    try std.testing.expectEqualStrings("pd", Freshness.day.param());
    try std.testing.expectEqualStrings("pw", Freshness.week.param());
    try std.testing.expectEqualStrings("pm", Freshness.month.param());
    try std.testing.expectEqualStrings("py", Freshness.year.param());
}

const Fake = @import("fake.zig").Fake;

test "search and fetch decode the v1 answers, ignoring fields they do not know" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var fake: Fake = try .init(io, &.{
        .{ .status = 200, .body =
        \\{"query":"zig http","provider":"brave","new_field":1,"results":[
        \\{"title":"Zig","url":"https://ziglang.org/","snippet":"A language","snippets":["one","two"],"published":"2026-09-01"},
        \\{"title":"zig","url":"https://github.com/ziglang/zig","snippet":"","published":null}],
        \\"providers_failed":["parallel"]}
        },
        .{ .status = 200, .body =
        \\{"url":"https://example.org/a","final_url":"https://example.org/a/","provider":"local","providers_failed":[],"status":200,"content_type":"text/html",
        \\"title":"A","published":null,"text":"# A\n\nbody","start":0,"chars":48211,"truncated":true}
        },
        .{ .status = 200, .body =
        \\{"url":"https://example.org/m.pdf","final_url":"https://example.org/m.pdf","provider":"parallel","providers_failed":[],"status":null,"content_type":null,
        \\"title":"M","published":"2026-01-02","focused":true,"text":"the answer","start":0,"chars":10,"truncated":false}
        },
    });
    var task = try io.concurrent(Fake.serve, .{ &fake, io });
    defer fake.stop(io, &task);
    var buffer: [64]u8 = undefined;
    var service: Service = .init(a, io, fake.base(&buffer), "nuclis/test");
    defer service.deinit();

    var found = try service.search(a, "zig http&more", 8, null, null);
    defer found.deinit();
    const s = found.outcome.ok;
    try std.testing.expectEqualStrings("brave", s.provider);
    try std.testing.expectEqual(@as(usize, 2), s.results.len);
    try std.testing.expectEqualStrings("2026-09-01", s.results[0].published.?);
    try std.testing.expectEqualStrings("two", s.results[0].snippets[1]);
    try std.testing.expect(s.results[1].published == null);
    try std.testing.expectEqualStrings("parallel", s.providers_failed[0]);
    try std.testing.expectEqualStrings("/v1/search?q=zig%20http%26more&n=8", fake.target(0));
    try std.testing.expectEqualStrings("nuclis/test", fake.agent[0..fake.agent_len]);

    var page = try service.fetch(a, "https://example.org/a?b=1", 100, 2000, null, null);
    defer page.deinit();
    const f = page.outcome.ok;
    try std.testing.expectEqual(@as(u64, 48211), f.chars);
    try std.testing.expect(f.truncated);
    try std.testing.expect(!f.focused);
    try std.testing.expectEqual(@as(?u16, 200), f.status);
    try std.testing.expectEqualStrings("text/html", f.content_type.?);
    try std.testing.expectEqualStrings("# A\n\nbody", f.text);
    try std.testing.expectEqualStrings("/v1/fetch?url=https%3A%2F%2Fexample.org%2Fa%3Fb%3D1&start=100&max_chars=2000", fake.target(1));

    // Parallel reports neither status nor content type.
    var asked = try service.fetch(a, "https://example.org/m.pdf", 0, 16000, "what is new?", null);
    defer asked.deinit();
    const m = asked.outcome.ok;
    try std.testing.expect(m.status == null and m.content_type == null);
    try std.testing.expect(m.focused);
    try std.testing.expectEqualStrings("parallel", m.provider);
    try std.testing.expectEqualStrings("2026-01-02", m.published.?);
    try std.testing.expectEqualStrings("/v1/fetch?url=https%3A%2F%2Fexample.org%2Fm.pdf&start=0&max_chars=16000&q=what%20is%20new%3F", fake.target(2));
}

test "the service's error codes, a proxy's page, and a malformed answer are failures" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    const busy_body = "{\"error\":{\"code\":\"busy\",\"message\":\"every slot stayed taken for 2 s\"}}";
    var fake: Fake = try .init(io, &.{
        .{ .status = 503, .body = busy_body },
        .{ .status = 503, .body = busy_body },
        .{ .status = 502, .body = "<html>Bad Gateway</html>" },
        .{ .status = 200, .body = "{\"results\": 7}" },
        .{ .status = 200, .body = "{\"version\":\"0.2.0\",\"providers\":{\"parallel\":\"down\",\"brave\":\"down\"}}" },
    });
    var task = try io.concurrent(Fake.serve, .{ &fake, io });
    defer fake.stop(io, &task);
    var buffer: [64]u8 = undefined;
    var service: Service = .init(a, io, fake.base(&buffer), "nuclis/test");
    defer service.deinit();
    service.retry_pause_ms = 1;

    var busy = try service.search(a, "q", 8, null, null);
    defer busy.deinit();
    try std.testing.expectEqualStrings("busy", busy.outcome.failed.code);
    try std.testing.expectEqual(@as(u16, 503), busy.outcome.failed.status);
    var proxy = try service.search(a, "q", 8, null, null);
    defer proxy.deinit();
    try std.testing.expectEqualStrings("invalid_response", proxy.outcome.failed.code);
    try std.testing.expect(std.mem.indexOf(u8, proxy.outcome.failed.message, "502") != null);
    var malformed = try service.search(a, "q", 8, null, null);
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
    const late = try service.call(Health, arena.allocator(), url, 100, false, null);
    try std.testing.expectEqualStrings("timeout", late.failed.code);
    // A timeout is not retried: one request reached the service.
    try std.testing.expectEqual(@as(usize, 1), fake.served.load(.acquire));

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

test "context asks for the fixed budget and decodes sources; search decodes snippets and the corrected query" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var fake: Fake = try .init(io, &.{
        .{ .status = 200, .body =
        \\{"query":"zig usingnamespace","provider":"brave","new_field":true,"sources":[
        \\{"url":"https://ziglang.org/download/0.15.1/release-notes.html","title":"0.15.1 Release Notes","published":"2025-08-19","snippets":["usingnamespace is removed.","Use a field instead."],"rank":1},
        \\{"url":"https://example.org/z"}],"providers_failed":[]}
        },
        .{ .status = 200, .body =
        \\{"query":"zgi http","altered_query":"zig http","provider":"brave","results":[
        \\{"title":"Zig","url":"https://ziglang.org/","snippet":"A language","snippets":["one","two","three"],"published":null}],"providers_failed":[]}
        },
    });
    var task = try io.concurrent(Fake.serve, .{ &fake, io });
    defer fake.stop(io, &task);
    var buffer: [64]u8 = undefined;
    var service: Service = .init(a, io, fake.base(&buffer), "nuclis/test");
    defer service.deinit();

    var passages = try service.context(a, "zig usingnamespace", .week, null);
    defer passages.deinit();
    const c = passages.outcome.ok;
    try std.testing.expectEqualStrings("/v1/context?q=zig%20usingnamespace&tokens=2048&urls=8&freshness=pw", fake.target(0));
    try std.testing.expectEqual(@as(usize, 2), c.sources.len);
    try std.testing.expectEqualStrings("2025-08-19", c.sources[0].published.?);
    try std.testing.expectEqualStrings("Use a field instead.", c.sources[0].snippets[1]);
    try std.testing.expectEqualStrings("", c.sources[1].title);
    try std.testing.expectEqual(@as(usize, 0), c.sources[1].snippets.len);

    var found = try service.search(a, "zgi http", 8, null, null);
    defer found.deinit();
    const s = found.outcome.ok;
    try std.testing.expectEqualStrings("/v1/search?q=zgi%20http&n=8", fake.target(1));
    try std.testing.expectEqualStrings("zig http", s.altered_query.?);
    try std.testing.expectEqual(@as(usize, 3), s.results[0].snippets.len);
}

test "the probe offers web_context only when health lists it" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var fake: Fake = try .init(io, &.{
        .{ .status = 200, .body = "{\"version\":\"0.5.0\",\"providers\":{\"brave\":\"ok\",\"local\":\"ok\",\"parallel\":\"ok\"},\"routes\":{\"/v1/search\":[\"parallel\",\"brave\"]},\"endpoints\":[\"/v1/search\",\"/v1/fetch\",\"/v1/context\"],\"billed\":{\"brave\":0,\"parallel\":5}}" },
        .{ .status = 200, .body = "{\"version\":\"0.2.0\",\"providers\":{\"parallel\":\"ok\"}}" },
        .{ .status = 200, .body = "{}", .delay_ms = 2_500 },
    });
    var task = try io.concurrent(Fake.serve, .{ &fake, io });
    defer fake.stop(io, &task);
    var buffer: [64]u8 = undefined;
    var service: Service = .init(a, io, fake.base(&buffer), "nuclis/test");
    defer service.deinit();

    try service.probe(a);
    try std.testing.expect(service.context_offered);
    try service.probe(a);
    try std.testing.expect(!service.context_offered);
    service.context_offered = true;
    // Slower than the probe's 2 s: no context.
    try service.probe(a);
    try std.testing.expect(!service.context_offered);
}

test "the allowance: six uncached calls, then budget without a request; hits and plain fetches are free" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    const search_body = "{\"query\":\"q\",\"results\":[]}";
    const context_body = "{\"query\":\"q\",\"sources\":[]}";
    const fetch_body = "{\"url\":\"https://e.org/\",\"text\":\"x\",\"chars\":1}";
    var fake: Fake = try .init(io, &.{
        .{ .status = 200, .body = search_body, .cache = "hit" },
        .{ .status = 200, .body = search_body, .cache = "stale" },
        .{ .status = 200, .body = search_body, .cache = "miss" },
        .{ .status = 200, .body = context_body },
        .{ .status = 200, .body = search_body },
        .{ .status = 200, .body = search_body },
        .{ .status = 200, .body = context_body },
        .{ .status = 200, .body = fetch_body },
        .{ .status = 200, .body = search_body },
        .{ .status = 200, .body = search_body },
    });
    var task = try io.concurrent(Fake.serve, .{ &fake, io });
    defer fake.stop(io, &task);
    var buffer: [64]u8 = undefined;
    var service: Service = .init(a, io, fake.base(&buffer), "nuclis/test");
    defer service.deinit();

    for (0..2) |_| {
        var hit = try service.search(a, "q", 8, null, null);
        hit.deinit();
    }
    try std.testing.expectEqual(@as(u8, 0), service.billed);
    for (0..3) |_| {
        var miss = try service.search(a, "q", 8, null, null);
        miss.deinit();
        var passages = try service.context(a, "q", null, null);
        passages.deinit();
    }
    try std.testing.expectEqual(@as(u8, 6), service.billed);
    var page = try service.fetch(a, "https://e.org/", 0, 100, null, null);
    page.deinit();
    try std.testing.expectEqual(@as(u8, 6), service.billed);

    var refused = try service.search(a, "q", 8, null, null);
    defer refused.deinit();
    try std.testing.expectEqualStrings("budget", refused.outcome.failed.code);
    var refused_context = try service.context(a, "q", null, null);
    defer refused_context.deinit();
    try std.testing.expectEqualStrings("budget", refused_context.outcome.failed.code);
    var refused_question = try service.fetch(a, "https://e.org/", 0, 100, "why?", null);
    defer refused_question.deinit();
    try std.testing.expectEqualStrings("budget", refused_question.outcome.failed.code);
    // Neither reached the service: nine answers taken, the tenth waits.
    try std.testing.expectEqual(@as(usize, 9), fake.served.load(.acquire));

    service.beginTurn();
    var again = try service.search(a, "q", 8, null, null);
    defer again.deinit();
    try std.testing.expectEqual(@as(u8, 1), service.billed);
}

test "a pooled connection the peer closed before answering is retried once on a fresh one" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    const health_body = "{\"version\":\"0.5.0\",\"providers\":{\"brave\":\"ok\"}}";
    var fake: Fake = try .init(io, &.{
        .{ .body = health_body, .keep_alive = true },
        .{ .drop = true },
        .{ .body = "{\"query\":\"q\",\"results\":[{\"url\":\"https://ziglang.org/\"}]}" },
        .{ .body = health_body, .keep_alive = true },
        .{ .drop = true },
        .{ .drop = true },
    });
    var task = try io.concurrent(Fake.serve, .{ &fake, io });
    defer fake.stop(io, &task);
    var buffer: [64]u8 = undefined;
    var service: Service = .init(a, io, fake.base(&buffer), "nuclis/test");
    defer service.deinit();

    var health = try service.health(a, null);
    health.deinit();
    var found = try service.search(a, "q", 8, null, null);
    defer found.deinit();
    try std.testing.expectEqualStrings("https://ziglang.org/", found.outcome.ok.results[0].url);
    try std.testing.expectEqual(@as(usize, 3), fake.served.load(.acquire));
    try std.testing.expectEqual(@as(u8, 1), service.billed);

    // The fresh connection closing too is the second failure: no third try.
    var again = try service.health(a, null);
    again.deinit();
    var lost = try service.search(a, "q", 8, null, null);
    defer lost.deinit();
    try std.testing.expectEqualStrings("unreachable", lost.outcome.failed.code);
    try std.testing.expect(std.mem.indexOf(u8, lost.outcome.failed.message, "HttpConnectionClosing") != null);
    try std.testing.expectEqual(@as(usize, 6), fake.served.load(.acquire));
}

test "a failure is retried by its code: transient once after the pause, the rest at once" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    const timeout_body = "{\"error\":{\"code\":\"upstream_timeout\",\"message\":\"no provider answered within 10 s\"}}";
    var fake: Fake = try .init(io, &.{
        .{ .status = 503, .body = "{\"error\":{\"code\":\"provider_unavailable\",\"message\":\"every provider was skipped\"}}" },
        .{ .status = 200, .body = "{\"query\":\"q\",\"provider\":\"brave\",\"results\":[{\"url\":\"https://ziglang.org/\"}]}" },
        .{ .status = 504, .body = timeout_body },
        .{ .status = 504, .body = timeout_body },
        .{ .status = 403, .body = "{\"error\":{\"code\":\"blocked_address\",\"message\":\"192.168.1.1 is not a public address\"}}" },
        .{ .status = 502, .body = "{\"error\":{\"code\":\"upstream_failed\",\"message\":\"go.dev answered 404 Not Found\"}}" },
        .{ .status = 503, .body = "{\"error\":{\"code\":\"busy\",\"message\":\"every slot stayed taken\"}}" },
    });
    var task = try io.concurrent(Fake.serve, .{ &fake, io });
    defer fake.stop(io, &task);
    var buffer: [64]u8 = undefined;
    var service: Service = .init(a, io, fake.base(&buffer), "nuclis/test");
    defer service.deinit();
    service.retry_pause_ms = 1;

    var recovered = try service.search(a, "q", 8, null, null);
    defer recovered.deinit();
    try std.testing.expectEqualStrings("https://ziglang.org/", recovered.outcome.ok.results[0].url);
    try std.testing.expectEqual(@as(usize, 2), fake.served.load(.acquire));
    try std.testing.expectEqual(@as(u8, 1), service.billed);

    var late = try service.context(a, "q", null, null);
    defer late.deinit();
    try std.testing.expectEqualStrings("upstream_timeout", late.outcome.failed.code);
    try std.testing.expectEqual(@as(usize, 4), fake.served.load(.acquire));
    try std.testing.expectEqual(@as(u8, 2), service.billed);

    var blocked = try service.fetch(a, "http://192.168.1.1/", 0, 100, null, null);
    defer blocked.deinit();
    try std.testing.expectEqualStrings("blocked_address", blocked.outcome.failed.code);
    try std.testing.expectEqual(@as(usize, 5), fake.served.load(.acquire));
    var dead = try service.fetch(a, "https://go.dev/x", 0, 100, null, null);
    defer dead.deinit();
    try std.testing.expectEqualStrings("upstream_failed", dead.outcome.failed.code);
    try std.testing.expectEqualStrings("go.dev answered 404 Not Found", dead.outcome.failed.message);
    try std.testing.expectEqual(@as(usize, 6), fake.served.load(.acquire));

    // A beat that cancels during the pause stops the retry.
    service.retry_pause_ms = 1_000;
    const Cancel = struct {
        fn now(_: *anyopaque) bool {
            return true;
        }
    };
    var dummy: u8 = 0;
    var cancelled = try service.fetch(a, "https://e.org/", 0, 100, null, .{ .context = &dummy, .call = Cancel.now });
    defer cancelled.deinit();
    try std.testing.expectEqualStrings("cancelled", cancelled.outcome.failed.code);
    try std.testing.expectEqual(@as(usize, 7), fake.served.load(.acquire));
}

test "the retry table sorts every code the contract names" {
    for ([_][]const u8{ "busy", "upstream_timeout", "provider_unavailable" }) |code| try std.testing.expectEqual(Retry.Class.transient, Retry.classOf(code));
    for ([_][]const u8{ "blocked_address", "bad_request", "unsupported_content", "not_available", "timeout", "budget", "some_new_code" }) |code| try std.testing.expectEqual(Retry.Class.permanent, Retry.classOf(code));
    try std.testing.expectEqual(Retry.Class.page, Retry.classOf("upstream_failed"));
}
