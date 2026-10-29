//! `GET /v1/models` and `GET /v1/models/{id}`: every service's models in
//! OpenAI's shape, so an OpenAI client reads the ids, and a `nuclis` object
//! per entry that nuclis clients read. Services fill in what only they know;
//! the fields every kind shares (`schema_version`, `endpoints`, `pinned`,
//! `detail`, the modalities) are completed here. docs/guide/api.md.
const std = @import("std");
const http = @import("http.zig");
const errors = @import("errors.zig");
const router_mod = @import("router.zig");
const model_files = @import("../model.zig");

/// Raised when a `nuclis` object changes incompatibly.
pub const schema_version = 1;

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
    /// The names the server keeps open for its whole life (`serve.pin`,
    /// `--pin`). Borrowed.
    pins: []const []const u8 = &.{},

    pub fn deinit(self: *Models, gpa: std.mem.Allocator) void {
        self.sources.deinit(gpa);
    }

    pub fn add(self: *Models, gpa: std.mem.Allocator, source: Source) !void {
        try self.sources.append(gpa, source);
    }

    pub fn register(self: *Models, gpa: std.mem.Allocator, router: *router_mod.Router) !void {
        try router.addSubtree(gpa, .GET, router_mod.prefix ++ "/models", .{ .context = self, .handle = handle });
    }

    fn handle(context: *anyopaque, arena: std.mem.Allocator, io: std.Io, request: http.Request) http.Response {
        const self: *Models = @ptrCast(@alignCast(context));
        var entries: std.ArrayList(Entry) = .empty;
        for (self.sources.items) |source| source.list(source.context, arena, io, &entries) catch |err|
            return .fromError(arena, .init(.internal_server_error, "internal", @errorName(err)));
        for (entries.items) |*e| complete(arena, e, self.pins) catch return .fromError(arena, .init(.internal_server_error, "internal", "out of memory"));
        const base = router_mod.prefix ++ "/models";
        var out: std.Io.Writer.Allocating = .init(arena);
        if (request.path.len > base.len) {
            const id = request.path[base.len + 1 ..];
            request.fields.text("model", id);
            const entry = for (entries.items) |*e| {
                if (std.mem.eql(u8, e.id, id)) break e;
            } else return .fromError(arena, .init(.not_found, "model_not_found", arena.print("{s} is not a model of this server (GET {s} lists them)", .{ id, base }) catch "model not found"));
            writeOne(&out.writer, entry.*) catch return .fromError(arena, .init(.internal_server_error, "internal", "out of memory"));
        } else {
            request.fields.int("models", entries.items.len);
            write(&out.writer, entries.items) catch return .fromError(arena, .init(.internal_server_error, "internal", "out of memory"));
        }
        return conditional(arena, request, out.written());
    }
};

/// One request field a model takes, for a `nuclis` object's `options`:
/// its type, what it accepts, and what this server uses when a request
/// leaves it out. Unset bounds are left out of the JSON; `default` is not.
pub const Option = struct {
    type: Type,
    values: ?std.json.Array = null,
    min: ?std.json.Value = null,
    max: ?std.json.Value = null,
    max_items: ?usize = null,
    max_bytes: ?usize = null,
    default: std.json.Value = .null,
    /// A language model's default where an effort changes it.
    by_effort: ?std.json.ObjectMap = null,

    pub const Type = enum { @"enum", enum_list, number, integer, string, string_list, boolean };

    pub fn value(self: Option, arena: std.mem.Allocator) !std.json.Value {
        var o: std.json.ObjectMap = .empty;
        try o.put(arena, "type", .{ .string = @tagName(self.type) });
        if (self.values) |v| try o.put(arena, "values", .{ .array = v });
        if (self.min) |v| try o.put(arena, "min", v);
        if (self.max) |v| try o.put(arena, "max", v);
        if (self.max_items) |v| try o.put(arena, "max_items", .{ .integer = @intCast(v) });
        if (self.max_bytes) |v| try o.put(arena, "max_bytes", .{ .integer = @intCast(v) });
        try o.put(arena, "default", self.default);
        if (self.by_effort) |b| if (b.count() > 0) try o.put(arena, "by_effort", .{ .object = b });
        return .{ .object = o };
    }
};

/// A float as the shortest decimal that reads back as `x` (`0.95`, not the
/// f32's `0.949999988`).
pub fn number(arena: std.mem.Allocator, x: f32) !std.json.Value {
    return .{ .number_string = try arena.print("{d}", .{x}) };
}

/// String values as a JSON array.
pub fn strings(arena: std.mem.Allocator, items: []const []const u8) !std.json.Array {
    var out: std.json.Array = .init(arena);
    for (items) |i| try out.append(.{ .string = i });
    return out;
}

/// Integer values as a JSON array.
pub fn integers(arena: std.mem.Allocator, items: anytype) !std.json.Array {
    var out: std.json.Array = .init(arena);
    for (items) |i| try out.append(.{ .integer = @intCast(i) });
    return out;
}

/// The body with a strong ETag of its bytes, or an empty `304` when the
/// client's `If-None-Match` already holds it.
fn conditional(arena: std.mem.Allocator, request: http.Request, body: []const u8) http.Response {
    const tag = arena.print("\"{x:0>16}\"", .{std.hash.Wyhash.hash(0, body)}) catch return .{ .body = body };
    const header: std.http.Header = .{ .name = "etag", .value = tag };
    if (request.if_none_match) |held| if (matchesTag(held, tag)) return .{ .status = .not_modified, .body = "", .extra_header = header };
    return .{ .body = body, .extra_header = header };
}

/// `If-None-Match`'s comparison is weak: `W/"x"` matches `"x"`, and `*` any.
fn matchesTag(held: []const u8, tag: []const u8) bool {
    var it = std.mem.tokenizeAny(u8, held, ", \t");
    while (it.next()) |candidate| {
        if (std.mem.eql(u8, candidate, "*")) return true;
        const bare = if (std.mem.startsWith(u8, candidate, "W/")) candidate[2..] else candidate;
        if (std.mem.eql(u8, bare, tag)) return true;
    }
    return false;
}

/// The routes that serve a model of `kind`.
fn endpoints(kind: []const u8) []const []const u8 {
    const table = .{
        .{ "decision", &[_][]const u8{ "/v1/decisions", "/v1/systemone" } },
        .{ "language", &[_][]const u8{"/v1/chat/completions"} },
        .{ "embedding", &[_][]const u8{"/v1/embeddings"} },
        .{ "speech", &[_][]const u8{"/v1/audio/speech"} },
        .{ "transcription", &[_][]const u8{ "/v1/audio/transcriptions", "/v1/audio/transcriptions/stream" } },
    };
    inline for (table) |row| if (std.mem.eql(u8, kind, row[0])) return row[1];
    return &.{};
}

/// Adds what every kind shares to a service's `nuclis` object:
/// `schema_version` first, then the service's fields, then `endpoints`,
/// `pinned`, the modalities, and a `detail` where the service gave none.
fn complete(arena: std.mem.Allocator, entry: *Entry, pins: []const []const u8) !void {
    const given = entry.details.object;
    var details: std.json.ObjectMap = .empty;
    try details.put(arena, "schema_version", .{ .integer = schema_version });
    var it = given.iterator();
    while (it.next()) |kv| try details.put(arena, kv.key_ptr.*, kv.value_ptr.*);
    const kind = if (given.get("kind")) |k| k.string else "";
    var routes: std.json.Array = .init(arena);
    for (endpoints(kind)) |r| try routes.append(.{ .string = r });
    try details.put(arena, "endpoints", .{ .array = routes });
    var pinned = false;
    for (pins) |p| pinned = pinned or std.mem.eql(u8, p, entry.id);
    try details.put(arena, "pinned", .{ .bool = pinned });
    const flag = struct {
        fn of(d: std.json.ObjectMap, name: []const u8) bool {
            return if (d.get(name)) |v| v == .bool and v.bool else false;
        }
    }.of;
    var inputs: std.json.Array = .init(arena);
    var outputs: std.json.Array = .init(arena);
    if (std.mem.eql(u8, kind, "embedding")) {
        if (given.get("modalities")) |m| inputs = m.array;
        try outputs.append(.{ .string = "embedding" });
    } else if (std.mem.eql(u8, kind, "speech")) {
        try inputs.append(.{ .string = "text" });
        try outputs.append(.{ .string = "audio" });
    } else if (std.mem.eql(u8, kind, "transcription")) {
        try inputs.append(.{ .string = "audio" });
        try outputs.append(.{ .string = "text" });
    } else {
        try inputs.append(.{ .string = "text" });
        if (flag(given, "images")) try inputs.append(.{ .string = "image" });
        if (flag(given, "audio")) try inputs.append(.{ .string = "audio" });
        try outputs.append(.{ .string = if (std.mem.eql(u8, kind, "decision")) "decision" else "text" });
    }
    try details.put(arena, "input_modalities", .{ .array = inputs });
    try details.put(arena, "output_modalities", .{ .array = outputs });
    if (given.get("detail") == null) try details.put(arena, "detail", .{ .string = try detailOf(arena, given) });
    entry.details = .{ .object = details };
}

/// `architecture · quantization · size`, leaving out what is unknown.
fn detailOf(arena: std.mem.Allocator, d: std.json.ObjectMap) ![]const u8 {
    var parts: std.ArrayList([]const u8) = .empty;
    if (d.get("architecture")) |v| if (v == .string) try parts.append(arena, v.string);
    if (d.get("quantization")) |v| if (v == .string) try parts.append(arena, v.string);
    var buffer: [16]u8 = undefined;
    if (d.get("size_bytes")) |v| if (v == .integer and v.integer > 0) try parts.append(arena, try arena.dupe(u8, model_files.humanSize(&buffer, @intCast(v.integer))));
    return std.mem.join(arena, " · ", parts.items);
}

pub fn write(out: *std.Io.Writer, entries: []const Entry) std.Io.Writer.Error!void {
    var s: std.json.Stringify = .{ .writer = out, .options = .{ .whitespace = .indent_2 } };
    try s.beginObject();
    try s.objectField("object");
    try s.write("list");
    try s.objectField("data");
    try s.beginArray();
    for (entries) |e| try writeEntry(&s, e);
    try s.endArray();
    try s.endObject();
    try out.writeByte('\n');
}

/// One entry alone: OpenAI's retrieve-model shape.
pub fn writeOne(out: *std.Io.Writer, entry: Entry) std.Io.Writer.Error!void {
    var s: std.json.Stringify = .{ .writer = out, .options = .{ .whitespace = .indent_2 } };
    try writeEntry(&s, entry);
    try out.writeByte('\n');
}

fn writeEntry(s: *std.json.Stringify, e: Entry) std.Io.Writer.Error!void {
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

test "every kind gets the shared fields; the modalities follow the kind" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var language: std.json.ObjectMap = .empty;
    try language.put(arena, "kind", .{ .string = "language" });
    try language.put(arena, "images", .{ .bool = true });
    try language.put(arena, "audio", .{ .bool = false });
    try language.put(arena, "architecture", .{ .string = "gemma4" });
    try language.put(arena, "size_bytes", .{ .integer = 6_716_356_800 });
    var entry: Entry = .{ .id = "gemma", .owned_by = "nuclis", .details = .{ .object = language } };
    try complete(arena, &entry, &.{"gemma"});
    const d = entry.details.object;
    try std.testing.expectEqualStrings("schema_version", d.keys()[0]);
    try std.testing.expect(d.get("pinned").?.bool);
    try std.testing.expectEqualStrings("/v1/chat/completions", d.get("endpoints").?.array.items[0].string);
    try std.testing.expectEqual(@as(usize, 2), d.get("input_modalities").?.array.items.len);
    try std.testing.expectEqualStrings("text", d.get("output_modalities").?.array.items[0].string);
    try std.testing.expectEqualStrings("gemma4 · 6.72 GB", d.get("detail").?.string);

    var speech: std.json.ObjectMap = .empty;
    try speech.put(arena, "kind", .{ .string = "speech" });
    try speech.put(arena, "detail", .{ .string = "kept" });
    var voice: Entry = .{ .id = "tts", .owned_by = "Qwen", .details = .{ .object = speech } };
    try complete(arena, &voice, &.{"gemma"});
    try std.testing.expect(!voice.details.object.get("pinned").?.bool);
    try std.testing.expectEqualStrings("audio", voice.details.object.get("output_modalities").?.array.items[0].string);
    try std.testing.expectEqualStrings("kept", voice.details.object.get("detail").?.string);
}

test "a body's ETag answers a matching If-None-Match with an empty 304" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const plain: http.Request = .{ .method = .GET, .path = "/v1/models", .query = "", .body = "" };
    const first = conditional(arena, plain, "{}");
    try std.testing.expectEqual(std.http.Status.ok, first.status);
    const tag = first.extra_header.?.value;
    for ([_][]const u8{ tag, try arena.print("W/{s}", .{tag}), try arena.print("\"x\", {s}", .{tag}), "*" }) |held| {
        var again = plain;
        again.if_none_match = held;
        const cached = conditional(arena, again, "{}");
        try std.testing.expectEqual(std.http.Status.not_modified, cached.status);
        try std.testing.expectEqualStrings("", cached.body);
    }
    var stale = plain;
    stale.if_none_match = "\"0\"";
    try std.testing.expectEqual(std.http.Status.ok, conditional(arena, stale, "{}").status);
    try std.testing.expect(!std.mem.eql(u8, tag, conditional(arena, plain, "[]").extra_header.?.value));
}

test "an option leaves out unset bounds, keeps a null default, and prints floats short" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var by_effort: std.json.ObjectMap = .empty;
    try by_effort.put(arena, "off", try number(arena, 0.8));
    const top_p = try (Option{ .type = .number, .min = try number(arena, 0), .max = try number(arena, 1), .default = try number(arena, 0.95), .by_effort = by_effort }).value(arena);
    var out: std.Io.Writer.Allocating = .init(arena);
    try std.json.Stringify.value(top_p, .{}, &out.writer);
    try std.testing.expectEqualStrings("{\"type\":\"number\",\"min\":0,\"max\":1,\"default\":0.95,\"by_effort\":{\"off\":0.8}}", out.written());
    const seed = try (Option{ .type = .integer }).value(arena);
    out.clearRetainingCapacity();
    try std.json.Stringify.value(seed, .{}, &out.writer);
    try std.testing.expectEqualStrings("{\"type\":\"integer\",\"default\":null}", out.written());
}
