//! `GET /v1/models`: every service's models in OpenAI's list shape, so an
//! OpenAI client reads the ids, and a `nuclis` object per entry that only
//! nuclis clients read (kind, presence, whether it is open, budgets).
const std = @import("std");
const http = @import("http.zig");
const errors = @import("errors.zig");
const router_mod = @import("router.zig");

pub const Entry = struct {
    id: []const u8,
    owned_by: []const u8,
    /// The `nuclis` object; built in the request's arena.
    details: std.json.Value,
};

/// A service's contribution to the listing.
pub const Source = struct {
    context: *anyopaque,
    list: *const fn (context: *anyopaque, arena: std.mem.Allocator, io: std.Io, out: *std.ArrayList(Entry)) anyerror!void,
};

pub const Models = struct {
    sources: std.ArrayList(Source) = .empty,

    pub fn deinit(self: *Models, gpa: std.mem.Allocator) void {
        self.sources.deinit(gpa);
    }

    pub fn add(self: *Models, gpa: std.mem.Allocator, source: Source) !void {
        try self.sources.append(gpa, source);
    }

    pub fn register(self: *Models, gpa: std.mem.Allocator, router: *router_mod.Router) !void {
        try router.add(gpa, .GET, router_mod.prefix ++ "/models", .{ .context = self, .handle = handle });
    }

    fn handle(context: *anyopaque, arena: std.mem.Allocator, io: std.Io, request: http.Request) http.Response {
        _ = request;
        const self: *Models = @ptrCast(@alignCast(context));
        var entries: std.ArrayList(Entry) = .empty;
        for (self.sources.items) |source| source.list(source.context, arena, io, &entries) catch |err|
            return .fromError(arena, .init(.internal_server_error, "internal", @errorName(err)));
        var out: std.Io.Writer.Allocating = .init(arena);
        write(&out.writer, entries.items) catch return .fromError(arena, .init(.internal_server_error, "internal", "out of memory"));
        return .{ .body = out.written() };
    }
};

pub fn write(out: *std.Io.Writer, entries: []const Entry) std.Io.Writer.Error!void {
    var s: std.json.Stringify = .{ .writer = out, .options = .{ .whitespace = .indent_2 } };
    try s.beginObject();
    try s.objectField("object");
    try s.write("list");
    try s.objectField("data");
    try s.beginArray();
    for (entries) |e| {
        try s.beginObject();
        try s.objectField("id");
        try s.write(e.id);
        try s.objectField("object");
        try s.write("model");
        // OpenAI's schema requires it; nuclis has no creation time to give.
        try s.objectField("created");
        try s.write(0);
        try s.objectField("owned_by");
        try s.write(e.owned_by);
        try s.objectField("nuclis");
        try s.write(e.details);
        try s.endObject();
    }
    try s.endArray();
    try s.endObject();
    try out.writeByte('\n');
}

test "the listing is OpenAI's shape with a nuclis object" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var details: std.json.ObjectMap = .empty;
    try details.put(arena, "kind", .{ .string = "decision" });
    var out: std.Io.Writer.Allocating = .init(arena);
    try write(&out.writer, &.{.{ .id = "laya", .owned_by = "convaiinnovations", .details = .{ .object = details } }});
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, arena, out.written(), .{});
    try std.testing.expectEqualStrings("list", parsed.object.get("object").?.string);
    const first = parsed.object.get("data").?.array.items[0].object;
    try std.testing.expectEqualStrings("laya", first.get("id").?.string);
    try std.testing.expectEqualStrings("model", first.get("object").?.string);
    try std.testing.expectEqualStrings("decision", first.get("nuclis").?.object.get("kind").?.string);
}
