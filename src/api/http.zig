//! The API's transport: one connection's HTTP/1.1 requests (keep-alive, no
//! pipelining) over any reader and writer, each request read whole into the
//! connection's arena, handed to a `Handler`, and answered with
//! `Content-Length` in one flush, or streamed in chunks when the response
//! carries a `Stream`. Knows no route and no model; transport
//! failures answer with the shared error body and close. The arena is reset
//! after every response, so nothing of a request outlives it. Every
//! response is logged when a log is given.
const std = @import("std");
const repeat = @import("../text.zig").repeat;
const errors = @import("errors.zig");
const log_mod = @import("log.zig");

/// TypeSafe's "overloaded", which Jev clients retry with backoff; no
/// standard code says "the queue is full, come back" as plainly.
pub const overloaded: std.http.Status = @fromBackingInt(529);

fn reason(status: std.http.Status) []const u8 {
    return if (status == overloaded) "Overloaded" else status.phrase() orelse "";
}

pub const Limits = struct {
    /// The request line and headers; the connection's read buffer must be at
    /// least this large.
    max_head: usize = 16 * 1024,
    max_body: usize = 4 * 1024 * 1024,
};

pub const Request = struct {
    method: std.http.Method,
    /// The target before any `?`.
    path: []const u8,
    /// After the `?`, empty when there is none.
    query: []const u8,
    body: []const u8,

    /// The value of `name` in the query (`a=1&b`: `b` is ""), or null.
    pub fn param(self: Request, name: []const u8) ?[]const u8 {
        var it = std.mem.splitScalar(u8, self.query, '&');
        while (it.next()) |pair| {
            const eq = std.mem.indexOfScalar(u8, pair, '=') orelse pair.len;
            if (std.mem.eql(u8, pair[0..eq], name)) return if (eq < pair.len) pair[eq + 1 ..] else "";
        }
        return null;
    }
};

pub const Response = struct {
    status: std.http.Status = .ok,
    body: []const u8,
    content_type: []const u8 = "application/json",
    /// Ask the transport to close the connection after this response.
    close: bool = false,
    /// One line for the log: what was done, or the error.
    note: []const u8 = "",
    /// Set for a body written as it is produced; `body` is then unused.
    stream: ?Stream = null,

    pub fn fromError(arena: std.mem.Allocator, err: errors.ApiError) Response {
        return .{ .status = err.status, .body = errors.body(arena, err), .note = noteOf(arena, err) };
    }
};

/// A response body written after the head, on the connection's task:
/// `transfer-encoding: chunked`, or for an HTTP/1.0 client the body until
/// the connection closes. The head is flushed before `write` runs. `write`
/// returns false when the stream failed, which closes the connection;
/// `context` must outlive it.
pub const Stream = struct {
    context: *anyopaque,
    write: *const fn (context: *anyopaque, io: std.Io, body: *Body) bool,
};

/// What a `Stream` writes through.
pub const Body = struct {
    inner: *std.http.BodyWriter,
    /// Bytes sent so far, for the log.
    bytes: usize = 0,

    /// Writes `bytes` and flushes them to the client: one chunk.
    pub fn send(self: *Body, bytes: []const u8) std.Io.Writer.Error!void {
        try self.inner.writer.writeAll(bytes);
        try self.inner.writer.flush();
        try self.inner.flush();
        self.bytes += bytes.len;
    }
};

/// What answers a request. `handle` never fails: an expected failure is an
/// error response; `arena` lives until the response is written.
pub const Handler = struct {
    context: *anyopaque,
    handle: *const fn (context: *anyopaque, arena: std.mem.Allocator, io: std.Io, request: Request) Response,
    /// The body limit for `path` when it differs from the transport's;
    /// asked before the body is read.
    body_limit: ?*const fn (context: *anyopaque, path: []const u8) ?usize = null,
};

pub fn noteOf(arena: std.mem.Allocator, err: errors.ApiError) []const u8 {
    return arena.print("{s}: {s}", .{ err.code, err.message }) catch err.code;
}

/// Serves requests from `in` until the client closes, asks to close, or a
/// transport failure ends the connection. `in`'s buffer holds the head, so
/// the head limit is the smaller of `limits.max_head` and that buffer.
pub fn serve(gpa: std.mem.Allocator, io: std.Io, in: *std.Io.Reader, out: *std.Io.Writer, handler: Handler, limits: Limits, log: ?*log_mod.Log) void {
    var server = std.http.Server.init(in, out);
    server.reader.max_head_len = @min(limits.max_head, in.buffer.len);
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    while (true) {
        // A large body's pages are returned; the common small request reuses
        // what the previous one allocated.
        _ = arena_state.reset(.{ .retain_with_limit = 256 * 1024 });
        const arena = arena_state.allocator();
        if (!serveOne(arena, io, &server, handler, limits, log)) return;
    }
}

/// One request and its response; false when the connection is done.
fn serveOne(arena: std.mem.Allocator, io: std.Io, server: *std.http.Server, handler: Handler, limits: Limits, log: ?*log_mod.Log) bool {
    var request = server.receiveHead() catch |err| {
        const refusal: ?errors.ApiError = switch (err) {
            error.HttpConnectionClosing, error.HttpRequestTruncated, error.ReadFailed => null,
            error.HttpHeadersOversize => .init(.request_header_fields_too_large, "headers_too_large", "the request line and headers exceed the limit"),
            error.HttpHeadersInvalid => .init(.bad_request, "bad_request", "malformed HTTP request"),
        };
        if (refusal) |r| {
            const body = errors.body(arena, r);
            writeClosing(server.out, r.status, body) catch {};
            if (log) |l| l.request(io, .{ .method = null, .path = "-", .status = r.status, .duration_ns = 0, .bytes_out = body.len, .note = noteOf(arena, r) });
        }
        return false;
    };
    var exchange: Exchange = .{ .request = &request, .arena = arena, .io = io, .log = log, .started = std.Io.Clock.awake.now(io), .method = request.head.method };
    // The head's strings die when the body is read: copy what is kept.
    const target = arena.dupe(u8, request.head.target) catch return exchange.fail(.init(.internal_server_error, "internal", "out of memory"));
    const question = std.mem.indexOfScalar(u8, target, '?');
    var parsed: Request = .{
        .method = request.head.method,
        .path = target[0 .. question orelse target.len],
        .query = if (question) |q| target[q + 1 ..] else "",
        .body = "",
    };
    exchange.path = parsed.path;
    if (request.head.transfer_compression != .identity)
        return exchange.fail(.init(.unsupported_media_type, "unsupported_encoding", "compressed request bodies are not accepted"));
    if (request.head.method.requestHasBody()) {
        const max_body = (if (handler.body_limit) |limit| limit(handler.context, parsed.path) else null) orelse limits.max_body;
        if (request.head.content_length) |length| if (length > max_body)
            return exchange.fail(.init(.payload_too_large, "payload_too_large", "the request body exceeds the limit"));
        // HTTP/1.1: a request with neither length nor chunking has no body.
        if (request.head.transfer_encoding == .none and request.head.content_length == null) request.head.content_length = 0;
        const body_reader = request.readerExpectContinue(&.{}) catch |err| return switch (err) {
            error.HttpExpectationFailed => blk: {
                // `respond` would refuse the expectation again.
                request.head.expect = null;
                break :blk exchange.fail(.init(.expectation_failed, "bad_request", "unsupported Expect header"));
            },
            error.WriteFailed => false,
        };
        parsed.body = body_reader.allocRemaining(arena, .limited(max_body)) catch |err| return switch (err) {
            error.StreamTooLong => exchange.fail(.init(.payload_too_large, "payload_too_large", "the request body exceeds the limit")),
            error.OutOfMemory => exchange.fail(.init(.internal_server_error, "internal", "out of memory")),
            error.ReadFailed => false,
        };
    }
    const response = handler.handle(handler.context, arena, io, parsed);
    return exchange.respond(response) and !response.close;
}

/// One request's response, written and logged.
const Exchange = struct {
    request: *std.http.Server.Request,
    arena: std.mem.Allocator,
    io: std.Io,
    log: ?*log_mod.Log,
    started: std.Io.Timestamp,
    method: std.http.Method,
    path: []const u8 = "?",

    /// An error response that ends the connection: the request may not
    /// have been read whole.
    fn fail(self: *Exchange, err: errors.ApiError) bool {
        var response: Response = .fromError(self.arena, err);
        response.close = true;
        _ = self.respond(response);
        return false;
    }

    /// Whether the connection stays open.
    fn respond(self: *Exchange, response: Response) bool {
        if (response.stream) |stream| return self.respondStream(response, stream);
        const request = self.request;
        const sent = if (request.respond(response.body, .{
            // An HTTP/1.0 client keeps the connection only when told so.
            .version = request.head.version,
            .status = response.status,
            .reason = reason(response.status),
            .keep_alive = !response.close,
            .extra_headers = &.{.{ .name = "content-type", .value = response.content_type }},
        })) |_| true else |_| false;
        if (self.log) |l| l.request(self.io, .{
            .method = self.method,
            .path = self.path,
            .status = response.status,
            .duration_ns = @intCast(@max(0, self.started.durationTo(std.Io.Clock.awake.now(self.io)).toNanoseconds())),
            .bytes_out = response.body.len,
            .note = response.note,
        });
        return sent and request.head.keep_alive and !response.close;
    }

    fn respondStream(self: *Exchange, response: Response, stream: Stream) bool {
        const request = self.request;
        // HTTP/1.0 has no chunked encoding: the end of the connection ends the body.
        const http10 = request.head.version == .@"HTTP/1.0";
        const keep_alive = !response.close and !http10 and request.head.keep_alive;
        var buffer: [4096]u8 = undefined;
        var bytes: usize = 0;
        const sent = if (request.respondStreaming(&buffer, .{ .respond_options = .{
            .version = request.head.version,
            .status = response.status,
            .reason = reason(response.status),
            .keep_alive = keep_alive,
            .transfer_encoding = if (http10) .none else null,
            .extra_headers = &.{
                .{ .name = "content-type", .value = response.content_type },
                .{ .name = "cache-control", .value = "no-cache" },
            },
        } })) |writer| blk: {
            var inner = writer;
            var body: Body = .{ .inner = &inner };
            defer bytes = body.bytes;
            inner.flush() catch break :blk false;
            if (!stream.write(stream.context, self.io, &body)) break :blk false;
            inner.end() catch break :blk false;
            break :blk true;
        } else |_| false;
        if (self.log) |l| l.request(self.io, .{
            .method = self.method,
            .path = self.path,
            .status = response.status,
            .duration_ns = @intCast(@max(0, self.started.durationTo(std.Io.Clock.awake.now(self.io)).toNanoseconds())),
            .bytes_out = bytes,
            .note = response.note,
        });
        return sent and keep_alive;
    }
};

/// A whole response with `connection: close`, outside any request: the
/// transport's refusals and the accept loop's `busy`.
pub fn writeClosing(out: *std.Io.Writer, status: std.http.Status, body: []const u8) std.Io.Writer.Error!void {
    try out.print("HTTP/1.1 {d} {s}\r\nconnection: close\r\ncontent-type: application/json\r\ncontent-length: {d}\r\n\r\n{s}", .{ @backingInt(status), reason(status), body.len, body });
    try out.flush();
}

// ----- tests: in-memory streams, a handler that echoes -----

const Echo = struct {
    calls: usize = 0,
    fn handle(context: *anyopaque, arena: std.mem.Allocator, io: std.Io, request: Request) Response {
        _ = io;
        const self: *Echo = @ptrCast(@alignCast(context));
        self.calls += 1;
        if (std.mem.eql(u8, request.path, "/stream"))
            return .{ .body = "", .content_type = "text/event-stream", .stream = .{ .context = self, .write = writeStream } };
        if (std.mem.eql(u8, request.path, "/broken"))
            return .{ .body = "", .content_type = "text/event-stream", .stream = .{ .context = self, .write = writeBroken } };
        const text = arena.print("{s} {s} q={s} explain={s} body={s}", .{ @tagName(request.method), request.path, request.query, request.param("explain") orelse "-", request.body }) catch unreachable;
        return .{ .body = text, .content_type = "text/plain" };
    }

    fn bodyLimit(context: *anyopaque, path: []const u8) ?usize {
        _ = context;
        return if (std.mem.eql(u8, path, "/big")) 1000 else null;
    }

    fn writeStream(context: *anyopaque, io: std.Io, body: *Body) bool {
        _ = context;
        _ = io;
        for ([_][]const u8{ "data: a\n\n", "data: bc\n\n", "data: def\n\n" }) |piece| body.send(piece) catch return false;
        return true;
    }

    fn writeBroken(context: *anyopaque, io: std.Io, body: *Body) bool {
        _ = context;
        _ = io;
        body.send("data: x\n\n") catch return false;
        return false;
    }
};

/// Serves `input` as a socket would deliver it: through a read buffer of
/// `limits.max_head` bytes, in pieces of at most 7.
fn roundTrip(input: []const u8, limits: Limits, echo: *Echo) ![]u8 {
    const read_buffer = try std.testing.allocator.alloc(u8, limits.max_head);
    defer std.testing.allocator.free(read_buffer);
    var in: std.testing.Reader = .init(read_buffer, &.{.{ .buffer = input }});
    in.artificial_limit = .limited(7);
    var buffer: [64 * 1024]u8 = undefined;
    var out: std.Io.Writer = .fixed(&buffer);
    serve(std.testing.allocator, std.testing.io, &in.interface, &out, .{ .context = echo, .handle = Echo.handle, .body_limit = Echo.bodyLimit }, limits, null);
    return std.testing.allocator.dupe(u8, out.buffered());
}

fn expectContains(haystack: []const u8, needles: []const []const u8) !void {
    for (needles) |n| if (std.mem.indexOf(u8, haystack, n) == null) {
        std.debug.print("missing {s} in:\n{s}\n", .{ n, haystack });
        return error.TestExpectedContains;
    };
}

test "keep-alive serves every request on the connection, bodies by length and by chunks" {
    var echo: Echo = .{};
    const input = "GET /v1/health?explain=1 HTTP/1.1\r\nhost: x\r\n\r\n" ++
        "POST /v1/decisions HTTP/1.1\r\ncontent-length: 5\r\n\r\nhello" ++
        "POST /a HTTP/1.1\r\ntransfer-encoding: chunked\r\n\r\n3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n" ++
        "POST /empty HTTP/1.1\r\nconnection: close\r\n\r\n";
    const output = try roundTrip(input, .{ .max_head = 1024 }, &echo);
    defer std.testing.allocator.free(output);
    try std.testing.expectEqual(@as(usize, 4), echo.calls);
    try expectContains(output, &.{ "GET /v1/health q=explain=1 explain=1 body=", "POST /v1/decisions q= explain=- body=hello", "POST /a q= explain=- body=abcde", "POST /empty q= explain=- body=", "content-type: text/plain", "connection: close" });
    try std.testing.expectEqual(@as(usize, 4), std.mem.count(u8, output, "HTTP/1.1 200 OK"));
}

test "oversized heads and bodies, malformed requests: an error body, then close" {
    const cases = [_]struct { input: []const u8, status: []const u8, code: []const u8 }{
        .{ .input = "GET /" ++ repeat("a", 300) ++ " HTTP/1.1\r\n\r\nGET / HTTP/1.1\r\n\r\n", .status = "431", .code = "headers_too_large" },
        .{ .input = "POST / HTTP/1.1\r\ncontent-length: 101\r\n\r\n" ++ repeat("x", 101) ++ "GET / HTTP/1.1\r\n\r\n", .status = "413", .code = "payload_too_large" },
        .{ .input = "POST / HTTP/1.1\r\ntransfer-encoding: chunked\r\n\r\n80\r\n" ++ repeat("x", 128) ++ "\r\n0\r\n\r\n", .status = "413", .code = "payload_too_large" },
        .{ .input = "BREW / HTTP/1.1\r\n\r\n", .status = "400", .code = "bad_request" },
        .{ .input = "GET / HTTP/1.1\r\n broken\r\n\r\n", .status = "400", .code = "bad_request" },
        .{ .input = "POST / HTTP/1.1\r\ncontent-encoding: gzip\r\ncontent-length: 1\r\n\r\nx", .status = "415", .code = "unsupported_encoding" },
        .{ .input = "POST / HTTP/1.1\r\nexpect: tea\r\ncontent-length: 1\r\n\r\nx", .status = "417", .code = "bad_request" },
    };
    for (cases) |case| {
        errdefer std.debug.print("case: {s}\n", .{case.input[0..@min(case.input.len, 60)]});
        var echo: Echo = .{};
        const output = try roundTrip(case.input, .{ .max_head = 256, .max_body = 100 }, &echo);
        defer std.testing.allocator.free(output);
        try std.testing.expectEqual(@as(usize, 0), echo.calls);
        try std.testing.expect(std.mem.startsWith(u8, output, "HTTP/1.1 "));
        try expectContains(output, &.{ case.status, case.code, "connection: close" });
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, output, "HTTP/1.1 "));
    }
}

test "an HTTP/1.0 client is answered in 1.0, kept alive only when it asks" {
    var echo: Echo = .{};
    const output = try roundTrip("GET /a HTTP/1.0\r\nconnection: keep-alive\r\n\r\nGET /b HTTP/1.0\r\n\r\nGET /c HTTP/1.0\r\n\r\n", .{ .max_head = 1024 }, &echo);
    defer std.testing.allocator.free(output);
    try std.testing.expectEqual(@as(usize, 2), echo.calls);
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, output, "HTTP/1.0 200 OK"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, output, "connection: keep-alive"));
}

test "a client that closes mid-head or between requests gets nothing more" {
    var echo: Echo = .{};
    const empty = try roundTrip("", .{ .max_head = 256 }, &echo);
    defer std.testing.allocator.free(empty);
    try std.testing.expectEqualStrings("", empty);
    const truncated = try roundTrip("GET / HTTP/1.1\r\nhost:", .{ .max_head = 256 }, &echo);
    defer std.testing.allocator.free(truncated);
    try std.testing.expectEqualStrings("", truncated);
    try std.testing.expectEqual(@as(usize, 0), echo.calls);
}

test "a streamed response goes out in chunks and the connection serves the next request" {
    var echo: Echo = .{};
    const output = try roundTrip("POST /stream HTTP/1.1\r\ncontent-length: 2\r\n\r\n{}GET /after HTTP/1.1\r\n\r\n", .{ .max_head = 1024 }, &echo);
    defer std.testing.allocator.free(output);
    try std.testing.expectEqual(@as(usize, 2), echo.calls);
    try expectContains(output, &.{ "transfer-encoding: chunked", "content-type: text/event-stream", "cache-control: no-cache", "data: a\n\n\r\n", "data: bc\n\n\r\n", "data: def\n\n\r\n", "0\r\n\r\n", "GET /after" });
    const head = output[0..std.mem.indexOf(u8, output, "\r\n\r\n").?];
    try std.testing.expect(std.mem.indexOf(u8, head, "content-length") == null);
}

test "a stream that fails midway ends the connection" {
    var echo: Echo = .{};
    const output = try roundTrip("POST /broken HTTP/1.1\r\ncontent-length: 0\r\n\r\nGET /after HTTP/1.1\r\n\r\n", .{ .max_head = 1024 }, &echo);
    defer std.testing.allocator.free(output);
    try std.testing.expectEqual(@as(usize, 1), echo.calls);
    try expectContains(output, &.{"data: x"});
    try std.testing.expect(std.mem.indexOf(u8, output, "GET /after") == null);
}

test "an HTTP/1.0 client gets a stream without chunks, then the connection closes" {
    var echo: Echo = .{};
    const output = try roundTrip("POST /stream HTTP/1.0\r\nconnection: keep-alive\r\ncontent-length: 0\r\n\r\nGET /after HTTP/1.0\r\n\r\n", .{ .max_head = 1024 }, &echo);
    defer std.testing.allocator.free(output);
    try std.testing.expectEqual(@as(usize, 1), echo.calls);
    try std.testing.expect(std.mem.indexOf(u8, output, "transfer-encoding") == null);
    try std.testing.expect(std.mem.endsWith(u8, output, "data: a\n\ndata: bc\n\ndata: def\n\n"));
}

test "a route's own body limit replaces the transport's" {
    var echo: Echo = .{};
    const body = repeat("x", 500);
    const output = try roundTrip("POST /big HTTP/1.1\r\ncontent-length: 500\r\n\r\n" ++ body ++ "POST /small HTTP/1.1\r\ncontent-length: 500\r\n\r\n" ++ body, .{ .max_head = 1024, .max_body = 100 }, &echo);
    defer std.testing.allocator.free(output);
    try std.testing.expectEqual(@as(usize, 1), echo.calls);
    try expectContains(output, &.{ "POST /big q= explain=- body=xxx", "413", "payload_too_large" });
}

test "query parameters" {
    const r: Request = .{ .method = .GET, .path = "/", .query = "a=1&explain&b=", .body = "" };
    try std.testing.expectEqualStrings("1", r.param("a").?);
    try std.testing.expectEqualStrings("", r.param("explain").?);
    try std.testing.expectEqualStrings("", r.param("b").?);
    try std.testing.expect(r.param("c") == null);
}
