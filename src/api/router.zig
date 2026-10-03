//! Method and path to a service's handler. Services register their routes
//! (full paths under `/v1`); an unknown path is 404 and a known path with
//! another method 405, both with the shared error body. Paths match exactly.
const std = @import("std");
const http = @import("http.zig");
const errors = @import("errors.zig");

pub const prefix = "/v1";

pub const Route = struct {
    method: std.http.Method,
    path: []const u8,
    handler: http.Handler,
};

pub const Router = struct {
    routes: std.ArrayList(Route) = .empty,

    pub fn deinit(self: *Router, gpa: std.mem.Allocator) void {
        self.routes.deinit(gpa);
    }

    /// Registers `path` (which starts with `prefix`) for `method`.
    pub fn add(self: *Router, gpa: std.mem.Allocator, method: std.http.Method, path: []const u8, route: http.Handler) !void {
        std.debug.assert(std.mem.startsWith(u8, path, prefix));
        for (self.routes.items) |r| std.debug.assert(!(r.method == method and std.mem.eql(u8, r.path, path)));
        try self.routes.append(gpa, .{ .method = method, .path = path, .handler = route });
    }

    /// The router as the transport's handler; `self` must outlive it.
    pub fn handler(self: *Router) http.Handler {
        return .{ .context = self, .handle = handle };
    }

    fn handle(context: *anyopaque, arena: std.mem.Allocator, io: std.Io, request: http.Request) http.Response {
        const self: *Router = @ptrCast(@alignCast(context));
        var path_known = false;
        for (self.routes.items) |r| {
            if (!std.mem.eql(u8, r.path, request.path)) continue;
            // HEAD is GET without the body; the transport drops the body.
            if (r.method == request.method or (r.method == .GET and request.method == .HEAD))
                return r.handler.handle(r.handler.context, arena, io, request);
            path_known = true;
        }
        if (path_known) return .fromError(arena, .init(.method_not_allowed, "method_not_allowed", arena.print("{s} does not take {s}", .{ request.path, @tagName(request.method) }) catch "method not allowed"));
        return .fromError(arena, .init(.not_found, "not_found", arena.print("no route {s} (the routes are under {s}, see GET {s}/models)", .{ request.path, prefix, prefix }) catch "not found"));
    }
};

const Stub = struct {
    fn ok(context: *anyopaque, arena: std.mem.Allocator, io: std.Io, request: http.Request) http.Response {
        _ = context;
        _ = io;
        return .{ .body = arena.print("{s}", .{request.path}) catch unreachable };
    }
};

test "routes: a match, 404, 405, and HEAD as GET" {
    const gpa = std.testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var router: Router = .{};
    defer router.deinit(gpa);
    var unused: u8 = 0;
    const stub: http.Handler = .{ .context = &unused, .handle = Stub.ok };
    try router.add(gpa, .POST, "/v1/decisions", stub);
    try router.add(gpa, .GET, "/v1/health", stub);
    const h = router.handler();
    const io = std.testing.io;

    const hit = h.handle(h.context, arena, io, .{ .method = .POST, .path = "/v1/decisions", .query = "", .body = "" });
    try std.testing.expectEqual(std.http.Status.ok, hit.status);
    try std.testing.expectEqualStrings("/v1/decisions", hit.body);
    const head = h.handle(h.context, arena, io, .{ .method = .HEAD, .path = "/v1/health", .query = "", .body = "" });
    try std.testing.expectEqual(std.http.Status.ok, head.status);

    const wrong = h.handle(h.context, arena, io, .{ .method = .GET, .path = "/v1/decisions", .query = "", .body = "" });
    try std.testing.expectEqual(std.http.Status.method_not_allowed, wrong.status);
    try std.testing.expect(std.mem.indexOf(u8, wrong.body, "\"code\":\"method_not_allowed\"") != null);
    const missing = h.handle(h.context, arena, io, .{ .method = .GET, .path = "/v1/nope", .query = "", .body = "" });
    try std.testing.expectEqual(std.http.Status.not_found, missing.status);
    try std.testing.expect(std.mem.indexOf(u8, missing.body, "\"code\":\"not_found\"") != null);
}
