//! Bounded HTTPS requests using injected Zig 0.16 Io on a shared, pooled
//! `std.http.Client`. Connections are keep-alive and reused across concurrent
//! tasks (the pool is mutex-protected; individual requests are per task), which
//! is what turns a download of thousands of ranges from a TLS handshake per
//! range into a handful of long-lived connections. Each response owns an
//! arena. No environment lookup, logging of URLs/tokens, or implicit redirect.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const Response = struct {
    arena: std.heap.ArenaAllocator,
    status: u16,
    headers: []const std.http.Header,
    body: []const u8,
    pub fn deinit(r: *Response) void {
        r.arena.deinit();
        r.* = undefined;
    }
    pub fn header(r: Response, name: []const u8) ?[]const u8 {
        for (r.headers) |h| if (std.ascii.eqlIgnoreCase(h.name, name)) return h.value;
        return null;
    }
};

pub const Options = struct {
    url: []const u8,
    token: ?[]const u8 = null,
    range: ?[]const u8 = null,
    head: bool = false,
    max_bytes: usize = 4 * 1024 * 1024,
    timeout_seconds: u32 = 120,
    /// When set, a 200/206 body is read into this caller-owned buffer, which
    /// it must fill exactly (Content-Length must match), and `Response.body`
    /// borrows it. Lets a range land in its place in a larger buffer.
    into: ?[]u8 = null,
};

/// A client whose connection pool outlives every request made through it. The
/// allocator must be thread safe (std.http.Client's requirement); the pool
/// keeps up to 32 idle connections per client.
pub fn client(gpa: Allocator, io: Io) std.http.Client {
    return .{ .allocator = gpa, .io = io, .read_buffer_size = 128 * 1024 };
}

pub fn validateUrl(url: []const u8) !std.Uri {
    if (url.len > 64_000 or std.mem.indexOfAny(u8, url, "\r\n\x00") != null) return error.InvalidUrl;
    const uri = std.Uri.parse(url) catch return error.InvalidUrl;
    if (!std.ascii.eqlIgnoreCase(uri.scheme, "https") or uri.host == null or uri.user != null or uri.password != null or uri.fragment != null)
        return error.InvalidUrl;
    return uri;
}

pub fn statusError(status: u16) !void {
    switch (status) {
        200, 206 => {},
        401 => return error.Unauthorized,
        403 => return error.Forbidden,
        404 => return error.NotFound,
        416 => return error.InvalidRange,
        429 => return error.RateLimited,
        500, 502, 503, 504 => return error.ServiceUnavailable,
        501 => return error.NotImplemented,
        else => return error.UnexpectedHttpStatus,
    }
}

/// One worker task owns the request and its allocations; the caller accesses
/// the response only after `Select` has joined the worker, so a timed-out
/// request is fully torn down (and its connection closed) before returning.
/// Deadlines require an Io implementation supporting concurrent tasks.
pub fn get(http: *std.http.Client, options: Options) !Response {
    const io = http.io;
    const Completion = union(enum) { request: anyerror!void, deadline: Io.Cancelable!void };
    var buffer: [2]Completion = undefined;
    var select = Io.Select(Completion).init(io, &buffer);
    var result: ?Response = null;
    defer {
        select.cancelDiscard(); // Join before freeing anything borrowed by the worker.
        if (result) |*r| r.deinit();
    }
    try select.concurrent(.request, run, .{ http, options, &result });
    try select.concurrent(.deadline, timeout, .{ io, options.timeout_seconds });
    switch (try select.await()) {
        .deadline => |r| {
            try r;
            return error.TimedOut;
        },
        .request => |r| try r,
    }
    const response = result orelse return error.InvalidResponse;
    result = null;
    return response;
}

fn timeout(io: Io, seconds: u32) Io.Cancelable!void {
    try io.sleep(.fromSeconds(seconds), .awake);
}

fn run(http: *std.http.Client, options: Options, result: *?Response) !void {
    const uri = try validateUrl(options.url);
    var arena = std.heap.ArenaAllocator.init(http.allocator);
    errdefer arena.deinit();
    const a = arena.allocator();
    var extra: [2]std.http.Header = undefined;
    var count: usize = 0;
    if (options.token) |token| {
        if (token.len == 0 or token.len > 64_000 or std.mem.indexOfAny(u8, token, "\r\n\x00") != null) return error.InvalidToken;
        extra[count] = .{ .name = "Authorization", .value = try std.fmt.allocPrint(a, "Bearer {s}", .{token}) };
        count += 1;
    }
    if (options.range) |range| {
        if (range.len > 64_000 or std.mem.indexOfAny(u8, range, "\r\n\x00") != null) return error.InvalidRange;
        extra[count] = .{ .name = "Range", .value = range };
        count += 1;
    }
    var req = try http.request(if (options.head) .HEAD else .GET, uri, .{
        .redirect_behavior = .unhandled,
        .extra_headers = extra[0..count],
        .headers = .{ .accept_encoding = .{ .override = "identity" } },
    });
    defer req.deinit();
    try req.sendBodiless();
    var res = try req.receiveHead(&.{});
    var headers: std.ArrayList(std.http.Header) = .empty;
    var iter = res.head.iterateHeaders();
    while (iter.next()) |h| try headers.append(a, .{ .name = try a.dupe(u8, h.name), .value = try a.dupe(u8, h.value) });
    const status: u16 = @intFromEnum(res.head.status);
    // Redirect bodies and error bodies are neither artifacts nor useful diagnostics.
    var body: []const u8 = &.{};
    if (!options.head and (status == 200 or status == 206)) {
        if (res.head.content_encoding != .identity) return error.UnsupportedContentEncoding;
        var transfer: [16 * 1024]u8 = undefined;
        const reader = res.reader(&transfer);
        if (options.into) |dest| {
            const n = res.head.content_length orelse return error.InvalidResponse;
            if (n != dest.len) return error.InvalidResponse;
            reader.readSliceAll(dest) catch |err| switch (err) {
                error.EndOfStream => return error.InvalidResponse,
                else => return err,
            };
            var probe: [1]u8 = undefined;
            if (try reader.readSliceShort(&probe) != 0) return error.InvalidResponse;
            result.* = .{ .arena = arena, .status = status, .headers = headers.items, .body = dest };
            return;
        }
        if (res.head.content_length) |n| if (n > options.max_bytes) return error.ResponseTooLarge;
        // Read to the end of the body (which is what lets the request mark its
        // connection reusable) into capacity reserved up front when the length
        // is known: xorb bodies reach 64 MiB, and a growing read would copy
        // them several times.
        var list: std.ArrayList(u8) = .empty;
        if (res.head.content_length) |n| try list.ensureTotalCapacityPrecise(a, @intCast(n + 1));
        reader.appendRemaining(a, &list, .limited(options.max_bytes + 1)) catch |err| switch (err) {
            error.StreamTooLong => return error.ResponseTooLarge,
            else => return err,
        };
        if (list.items.len > options.max_bytes) return error.ResponseTooLarge;
        if (res.head.content_length) |n| if (list.items.len != n) return error.InvalidResponse;
        body = list.items;
    }
    result.* = .{ .arena = arena, .status = status, .headers = headers.items, .body = body };
}

test "HTTPS URL validation rejects credential and transport confusion" {
    for ([_][]const u8{ "http://huggingface.co/a", "https://user@huggingface.co/a", "https://huggingface.co/a#frag", "https://huggingface.co/\r\n" }) |url|
        try std.testing.expectError(error.InvalidUrl, validateUrl(url));
    _ = try validateUrl("https://huggingface.co/a?x=1");
    try std.testing.expectError(error.Unauthorized, statusError(401));
    try std.testing.expectError(error.RateLimited, statusError(429));
}
