//! `/v1/audio/voices`: the saved voices a Qwen3-TTS Base speaks in, in
//! OpenAI's custom-voice shape. `POST` saves a clip (multipart
//! `audio_sample`, `name`, and nuclis's `reference_text`) as `nuclis voice
//! add` does, without opening a model: the first request in the voice
//! encodes it. `GET` lists them or returns one, `DELETE` removes one. The
//! store is `voices.zig`; docs/guide/api.md § Voices.
const std = @import("std");
const http = @import("../http.zig");
const errors = @import("../errors.zig");
const multipart = @import("../multipart.zig");
const router_mod = @import("../router.zig");
const paths = @import("../../paths.zig");
const voices = @import("../../voices.zig");

const ApiError = errors.ApiError;
const path = router_mod.prefix ++ "/audio/voices";

/// A clip's upload: a 30 s WAV at 48 kHz stereo 24-bit is 8.6 MB.
pub const max_body = 32 << 20;

pub const Routes = struct {
    /// The user root, borrowed; null refuses every request (no home).
    root: ?[]const u8,

    pub fn register(self: *Routes, gpa: std.mem.Allocator, router: *router_mod.Router) !void {
        try router.addLimited(gpa, .POST, path, .{ .context = self, .handle = create }, max_body);
        try router.addSubtree(gpa, .GET, path, .{ .context = self, .handle = get });
        try router.addSubtree(gpa, .DELETE, path, .{ .context = self, .handle = remove });
    }

    fn store(self: *Routes, arena: std.mem.Allocator, io: std.Io) ?voices.Store {
        const home = self.root orelse return null;
        return .{ .io = io, .path = paths.voicesPath(arena, home) catch return null };
    }

    fn create(context: *anyopaque, arena: std.mem.Allocator, io: std.Io, request: http.Request) http.Response {
        const self: *Routes = @ptrCast(@alignCast(context));
        const s = self.store(arena, io) orelse return noHome(arena);
        const form = multipart.parse(arena, request.body, request.content_type orelse "", .{}) catch |err| return refuse(arena, .bad_request, "invalid_request", null, "{s}", .{switch (err) {
            error.NotMultipart => "the body must be multipart/form-data, with the clip as the part named audio_sample",
            error.InvalidBoundary => "the multipart content type names no usable boundary",
            error.TooManyParts => "more than 32 form fields",
            error.HeaderTooLarge => "a form part's headers exceed 1 KiB",
            error.MissingName => "a form part has no content-disposition name",
            error.TooManyFiles => "more than one file part: send one clip",
            error.Malformed => "the multipart body is malformed or truncated",
            error.OutOfMemory => return internal(arena),
        }});
        const file = form.file() orelse return refuse(arena, .bad_request, "invalid_request", "audio_sample", "no clip: send it as the file part named audio_sample", .{});
        if (!std.mem.eql(u8, file.name, "audio_sample")) return refuse(arena, .bad_request, "invalid_request", "audio_sample", "the clip's part must be named audio_sample, not {s}", .{file.name});
        const name = std.mem.trim(u8, form.field("name") orelse return refuse(arena, .bad_request, "invalid_request", "name", "\"name\" is required: the voice's name, what a speech request's voice names", .{}), " \t\r\n");
        // `consent` (OpenAI's consent recording) has no meaning for a local clip and is ignored.
        const now = std.Io.Timestamp.now(io, .real).toSeconds();
        const added = s.add(arena, name, file.data, form.field("reference_text"), now) catch |err| return storeError(arena, err, name);
        return json(arena, .created, voiceObject(added));
    }

    fn get(context: *anyopaque, arena: std.mem.Allocator, io: std.Io, request: http.Request) http.Response {
        const self: *Routes = @ptrCast(@alignCast(context));
        const s = self.store(arena, io) orelse return noHome(arena);
        const listed = s.list(arena) catch return internal(arena);
        if (std.mem.eql(u8, request.path, path)) {
            const data = arena.alloc(Voice, listed.len) catch return internal(arena);
            for (data, listed) |*d, v| d.* = voiceObject(v);
            return json(arena, .ok, .{ .object = "list", .data = data });
        }
        const id = request.path[path.len + 1 ..];
        for (listed) |v| if (std.mem.eql(u8, v.name, id)) return json(arena, .ok, voiceObject(v));
        return refuse(arena, .not_found, "voice_not_found", "voice_id", "no saved voice named {s} (GET {s} lists them)", .{ id, path });
    }

    fn remove(context: *anyopaque, arena: std.mem.Allocator, io: std.Io, request: http.Request) http.Response {
        const self: *Routes = @ptrCast(@alignCast(context));
        const s = self.store(arena, io) orelse return noHome(arena);
        if (std.mem.eql(u8, request.path, path)) return refuse(arena, .bad_request, "invalid_request", "voice_id", "name the voice: DELETE {s}/<name>", .{path});
        const id = request.path[path.len + 1 ..];
        s.remove(id) catch |err| return storeError(arena, err, id);
        return json(arena, .ok, .{ .id = id, .object = "audio.voice.deleted", .deleted = true });
    }
};

/// OpenAI's voice object, with what nuclis keeps besides.
const Voice = struct {
    id: []const u8,
    object: []const u8 = "audio.voice",
    name: []const u8,
    created_at: i64,
    reference_text: ?[]const u8,
    seconds: f64,
};

fn voiceObject(v: voices.Summary) Voice {
    return .{ .id = v.name, .name = v.name, .created_at = v.created_at, .reference_text = v.reference_text, .seconds = v.seconds };
}

fn json(arena: std.mem.Allocator, status: std.http.Status, value: anytype) http.Response {
    var out: std.Io.Writer.Allocating = .init(arena);
    std.json.Stringify.value(value, .{}, &out.writer) catch return internal(arena);
    out.writer.writeByte('\n') catch return internal(arena);
    return .{ .status = status, .body = out.written() };
}

/// A store error as the client reads it.
fn storeError(arena: std.mem.Allocator, err: anyerror, name: []const u8) http.Response {
    var diag: @import("../../config.zig").Diagnostic = .{};
    voices.explain(err, name, "audio_sample", &diag);
    const message = if (diag.len > 0) diag.message() else @errorName(err);
    return switch (err) {
        error.InvalidVoiceName, error.ReservedVoiceName => refuse(arena, .bad_request, "invalid_request", "name", "{s}", .{message}),
        error.VoiceExists => refuse(arena, .conflict, "voice_exists", "name", "{s}", .{message}),
        error.VoiceNotFound => refuse(arena, .not_found, "voice_not_found", "voice_id", "{s}", .{message}),
        error.TooManyVoices => refuse(arena, .bad_request, "too_many_voices", null, "{s}", .{message}),
        error.ClipTooShort, error.ClipTooLong, error.UnsupportedAudioFormat, error.AudioTooLarge => refuse(arena, .bad_request, "invalid_request", "audio_sample", "{s}", .{message}),
        error.TranscriptTooLong, error.InvalidTranscript => refuse(arena, .bad_request, "invalid_request", "reference_text", "{s}", .{message}),
        error.OutOfMemory => internal(arena),
        else => refuse(arena, .internal_server_error, "voice_store_failed", null, "the voice store failed: {s}", .{message}),
    };
}

fn noHome(arena: std.mem.Allocator) http.Response {
    return refuse(arena, .internal_server_error, "no_home", null, "the server has no user root (HOME or NUCLIS_HOME) to keep voices in", .{});
}

fn refuse(arena: std.mem.Allocator, status: std.http.Status, code: []const u8, param: ?[]const u8, comptime format: []const u8, args: anytype) http.Response {
    var err: ApiError = .init(status, code, arena.print(format, args) catch code);
    err.param = param;
    return .fromError(arena, err);
}

fn internal(arena: std.mem.Allocator) http.Response {
    return .fromError(arena, .init(.internal_server_error, "internal", "out of memory"));
}

// ---- tests -----------------------------------------------------------------

/// A multipart body of `fields` and, when given, the clip as `audio_sample`.
fn testForm(arena: std.mem.Allocator, fields: []const [2][]const u8, clip: ?[]const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (fields) |f| try out.print(arena, "--b\r\ncontent-disposition: form-data; name=\"{s}\"\r\n\r\n{s}\r\n", .{ f[0], f[1] });
    if (clip) |bytes| {
        try out.print(arena, "--b\r\ncontent-disposition: form-data; name=\"audio_sample\"; filename=\"a.wav\"\r\ncontent-type: audio/wav\r\n\r\n", .{});
        try out.appendSlice(arena, bytes);
        try out.appendSlice(arena, "\r\n");
    }
    try out.appendSlice(arena, "--b--\r\n");
    return out.items;
}

fn tone(arena: std.mem.Allocator, seconds: f64) ![]u8 {
    const speak = @import("../../speak.zig");
    const n: usize = @intFromFloat(seconds * voices.sample_rate);
    const out = try arena.alloc(u8, 44 + 2 * n);
    out[0..44].* = try speak.wavHeader(n, voices.sample_rate);
    for (0..n) |i| std.mem.writeInt(i16, out[44 + 2 * i ..][0..2], speak.pcm16(@floatCast(0.2 * @sin(@as(f64, @floatFromInt(i)) / 20))), .little);
    return out;
}

test "voices: create, list, get, delete, and the refusals, as OpenAI's shapes" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var routes: Routes = .{ .root = try tmp.dir.realPathFileAlloc(io, ".", arena) };
    var router: router_mod.Router = .{};
    defer router.deinit(gpa);
    try routes.register(gpa, &router);
    const h = router.handler();
    const call = struct {
        fn run(handler: http.Handler, a: std.mem.Allocator, method: std.http.Method, target: []const u8, body: []const u8) http.Response {
            return handler.handle(handler.context, a, std.testing.io, .{ .method = method, .path = target, .query = "", .body = body, .content_type = "multipart/form-data; boundary=b" });
        }
    }.run;

    const clip = try tone(arena, 2);
    const made = call(h, arena, .POST, path, try testForm(arena, &.{ .{ "name", "anna" }, .{ "reference_text", "Hello there." }, .{ "consent", "cons_1" } }, clip));
    try std.testing.expectEqual(std.http.Status.created, made.status);
    const object = try std.json.parseFromSliceLeaky(std.json.Value, arena, made.body, .{});
    try std.testing.expectEqualStrings("anna", object.object.get("id").?.string);
    try std.testing.expectEqualStrings("audio.voice", object.object.get("object").?.string);
    try std.testing.expectEqualStrings("Hello there.", object.object.get("reference_text").?.string);
    const seconds = object.object.get("seconds").?;
    try std.testing.expectEqual(@as(f64, 2), if (seconds == .integer) @as(f64, @floatFromInt(seconds.integer)) else seconds.float);

    const listed = call(h, arena, .GET, path, "");
    const list = try std.json.parseFromSliceLeaky(std.json.Value, arena, listed.body, .{});
    try std.testing.expectEqualStrings("list", list.object.get("object").?.string);
    try std.testing.expectEqual(@as(usize, 1), list.object.get("data").?.array.items.len);
    try std.testing.expectEqual(std.http.Status.ok, call(h, arena, .GET, path ++ "/anna", "").status);

    const Case = struct { body: []const u8, status: std.http.Status, code: []const u8 };
    const cases = [_]Case{
        .{ .body = try testForm(arena, &.{.{ "name", "anna" }}, clip), .status = .conflict, .code = "voice_exists" },
        .{ .body = try testForm(arena, &.{.{ "name", "ryan" }}, clip), .status = .bad_request, .code = "invalid_request" },
        .{ .body = try testForm(arena, &.{.{ "name", "Bad Name" }}, clip), .status = .bad_request, .code = "invalid_request" },
        .{ .body = try testForm(arena, &.{.{ "name", "short" }}, try tone(arena, 0.5)), .status = .bad_request, .code = "invalid_request" },
        .{ .body = try testForm(arena, &.{.{ "name", "none" }}, null), .status = .bad_request, .code = "invalid_request" },
        .{ .body = try testForm(arena, &.{}, clip), .status = .bad_request, .code = "invalid_request" },
    };
    for (cases) |case| {
        const r = call(h, arena, .POST, path, case.body);
        try std.testing.expectEqual(case.status, r.status);
        try std.testing.expect(std.mem.indexOf(u8, r.body, case.code) != null);
    }
    try std.testing.expectEqual(std.http.Status.not_found, call(h, arena, .GET, path ++ "/nobody", "").status);
    const deleted = call(h, arena, .DELETE, path ++ "/anna", "");
    try std.testing.expect(std.mem.indexOf(u8, deleted.body, "\"deleted\":true") != null);
    try std.testing.expectEqual(std.http.Status.not_found, call(h, arena, .DELETE, path ++ "/anna", "").status);
    try std.testing.expectEqual(std.http.Status.bad_request, call(h, arena, .DELETE, path, "").status);
}
