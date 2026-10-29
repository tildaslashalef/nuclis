//! `POST /v1/audio/speech` on the wire: the request read and validated
//! (OpenAI's fields, nuclis's `seed` and `language`), and the response's
//! bytes: 16-bit PCM, a WAV header for a stream of unknown length, and the
//! server-sent events (`speech.audio.delta`, `speech.audio.done`).
//! What a field means and why some are refused: docs/guide/api.md § Speech.
const std = @import("std");
const errors = @import("../errors.zig");

const Allocator = std.mem.Allocator;
const Value = std.json.Value;

/// The longest `input`, in bytes: one or a few sentences a request.
pub const max_input_bytes = 16 * 1024;
pub const sample_rate = 24000;

pub const Format = enum { pcm, wav };
pub const StreamFormat = enum { audio, sse };

pub const Request = struct {
    /// A speech model's registry or catalogue name; null: the open one,
    /// else `speech.model`.
    model: ?[]const u8 = null,
    input: []const u8,
    /// A speaker of the model, or on a Base a saved voice; null:
    /// `speech.voice`.
    voice: ?[]const u8 = null,
    /// Given as OpenAI's custom voice, `{"id": …}`: a saved voice only.
    voice_by_id: bool = false,
    instructions: ?[]const u8 = null,
    format: Format = .wav,
    stream: StreamFormat = .audio,
    seed: ?u64 = null,
    /// A language of the model, or `auto`; null is `auto`.
    language: ?[]const u8 = null,
};

/// Why a request was refused: a 400 with this code, message, and field.
pub const Problem = struct {
    code: []const u8 = "invalid_request",
    message: []const u8 = "",
    param: ?[]const u8 = null,
};

pub const Error = error{ Refused, OutOfMemory };

/// Formats OpenAI names that this server does not encode.
const other_formats = [_][]const u8{ "mp3", "opus", "aac", "flac" };

/// Reads `body`. `Refused` sets `problem`; every slice lives in `arena`.
/// Fields OpenAI defines and nuclis does not read are ignored.
pub fn parse(arena: Allocator, body: []const u8, problem: *Problem) Error!Request {
    const root = std.json.parseFromSliceLeaky(Value, arena, body, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return refuse(arena, problem, "invalid_json", null, "the request body is not valid JSON ({s})", .{@errorName(err)}),
    };
    if (root != .object) return refuse(arena, problem, "invalid_request", "body", "the request body must be a JSON object", .{});
    const o = root.object;
    var request: Request = .{ .input = "" };
    request.model = try optionalString(arena, problem, o, "model");
    request.input = switch (o.get("input") orelse return refuse(arena, problem, "invalid_request", "input", "\"input\" is required", .{})) {
        .string => |s| s,
        else => return refuse(arena, problem, "invalid_request", "input", "\"input\" must be a string", .{}),
    };
    if (std.mem.trim(u8, request.input, " \t\r\n").len == 0) return refuse(arena, problem, "invalid_request", "input", "\"input\" is empty", .{});
    if (request.input.len > max_input_bytes)
        return refuse(arena, problem, "request_too_large", "input", "\"input\" is {d} bytes; a request speaks at most {d} (a few sentences: send a long text a sentence or two at a time)", .{ request.input.len, max_input_bytes });
    if (o.get("voice")) |v| switch (v) {
        .null => {},
        .string => |s| request.voice = s,
        .object => |custom| switch (custom.get("id") orelse .null) {
            .string => |id| {
                request.voice = id;
                request.voice_by_id = true;
            },
            else => return refuse(arena, problem, "invalid_request", "voice", "a custom \"voice\" is {{\"id\": \"<saved voice>\"}}", .{}),
        },
        else => return refuse(arena, problem, "invalid_request", "voice", "\"voice\" must be a string or {{\"id\": \"<saved voice>\"}}", .{}),
    };
    request.instructions = try optionalString(arena, problem, o, "instructions");
    if (request.instructions) |i| if (std.mem.trim(u8, i, " \t\r\n").len == 0) {
        request.instructions = null;
    };
    if (try optionalString(arena, problem, o, "response_format")) |f| {
        request.format = std.meta.stringToEnum(Format, f) orelse {
            for (other_formats) |other| if (std.mem.eql(u8, f, other))
                return refuse(arena, problem, "unsupported_feature", "response_format", "\"{s}\" is not encoded here; \"pcm\" (24 kHz 16-bit mono) or \"wav\"", .{f});
            return refuse(arena, problem, "invalid_request", "response_format", "\"response_format\" is \"pcm\" or \"wav\", not \"{s}\"", .{f});
        };
    }
    if (try optionalString(arena, problem, o, "stream_format")) |f| {
        request.stream = std.meta.stringToEnum(StreamFormat, f) orelse
            return refuse(arena, problem, "invalid_request", "stream_format", "\"stream_format\" is \"audio\" or \"sse\", not \"{s}\"", .{f});
    }
    if (o.get("speed")) |v| {
        const speed: f64 = switch (v) {
            .null => 1,
            .integer => |n| @floatFromInt(n),
            .float => |f| f,
            else => return refuse(arena, problem, "invalid_request", "speed", "\"speed\" must be a number", .{}),
        };
        if (speed != 1) return refuse(arena, problem, "unsupported_feature", "speed", "only \"speed\": 1 is spoken; the model has no rate control", .{});
    }
    if (o.get("seed")) |v| switch (v) {
        .null => {},
        .integer => |n| request.seed = std.math.cast(u64, n) orelse return refuse(arena, problem, "invalid_request", "seed", "\"seed\" must not be negative", .{}),
        else => return refuse(arena, problem, "invalid_request", "seed", "\"seed\" must be an integer", .{}),
    };
    request.language = try optionalString(arena, problem, o, "language");
    return request;
}

fn optionalString(arena: Allocator, problem: *Problem, o: std.json.ObjectMap, comptime field: []const u8) Error!?[]const u8 {
    return switch (o.get(field) orelse return null) {
        .null => null,
        .string => |s| s,
        else => refuse(arena, problem, "invalid_request", field, "\"" ++ field ++ "\" must be a string", .{}),
    };
}

fn refuse(arena: Allocator, problem: *Problem, code: []const u8, param: ?[]const u8, comptime format: []const u8, args: anytype) Error {
    problem.* = .{ .code = code, .param = param, .message = try std.fmt.allocPrint(arena, format, args) };
    return error.Refused;
}

pub fn contentType(request: Request) []const u8 {
    return switch (request.stream) {
        .sse => "text/event-stream",
        .audio => switch (request.format) {
            .pcm => "audio/pcm",
            .wav => "audio/wav",
        },
    };
}

/// A WAV header for 16-bit mono at `sample_rate` whose length is not known
/// when it is sent: both sizes are the largest a field holds, which players
/// read as "until the end of the stream".
pub fn streamingWavHeader() [44]u8 {
    var h: [44]u8 = undefined;
    @memcpy(h[0..4], "RIFF");
    std.mem.writeInt(u32, h[4..8], std.math.maxInt(u32), .little);
    @memcpy(h[8..16], "WAVEfmt ");
    std.mem.writeInt(u32, h[16..20], 16, .little);
    std.mem.writeInt(u16, h[20..22], 1, .little); // PCM
    std.mem.writeInt(u16, h[22..24], 1, .little); // mono
    std.mem.writeInt(u32, h[24..28], sample_rate, .little);
    std.mem.writeInt(u32, h[28..32], sample_rate * 2, .little);
    std.mem.writeInt(u16, h[32..34], 2, .little);
    std.mem.writeInt(u16, h[34..36], 16, .little);
    @memcpy(h[36..40], "data");
    std.mem.writeInt(u32, h[40..44], std.math.maxInt(u32), .little);
    return h;
}

/// `samples` in [−1, 1] as 16-bit little-endian PCM into `out`, which holds
/// two bytes a sample; returns the bytes written.
pub fn pcm16(samples: []const f32, out: []u8) []u8 {
    for (samples, 0..) |x, i| {
        const clamped = std.math.clamp(x, -1, 1);
        std.mem.writeInt(i16, out[2 * i ..][0..2], @intFromFloat(@round(clamped * 32767)), .little);
    }
    return out[0 .. 2 * samples.len];
}

/// The most bytes one chunk of `frames` frames (`samples_per_frame` each)
/// becomes on the wire, the WAV header included, framed as an event or not.
pub fn maxChunkBytes(frames: usize, samples_per_frame: usize, stream: StreamFormat) usize {
    const raw = frames * samples_per_frame * 2 + 44;
    return switch (stream) {
        .audio => raw,
        .sse => std.base64.standard.Encoder.calcSize(raw) + 64,
    };
}

/// One `speech.audio.delta` event carrying `audio` (base64).
pub fn writeDelta(out: *std.Io.Writer, audio: []const u8) std.Io.Writer.Error!void {
    try out.writeAll("data: {\"type\":\"speech.audio.delta\",\"audio\":\"");
    try std.base64.standard.Encoder.encodeWriter(out, audio);
    try out.writeAll("\"}\n\n");
}

pub const Usage = struct { input_tokens: usize, output_tokens: usize };

/// The `speech.audio.done` event: why the speech ended (nuclis's
/// `finish_reason`: `eos`, the model's end; `length`, the frame budget),
/// then the text's tokens in, the frames out.
pub fn writeDone(out: *std.Io.Writer, usage: Usage, finish: []const u8) std.Io.Writer.Error!void {
    try out.print("data: {{\"type\":\"speech.audio.done\",\"finish_reason\":\"{s}\",\"usage\":{{\"input_tokens\":{d},\"output_tokens\":{d},\"total_tokens\":{d}}}}}\n\n", .{ finish, usage.input_tokens, usage.output_tokens, usage.input_tokens + usage.output_tokens });
}

/// An error after the stream began: one `data:` line with the error body.
pub fn writeStreamError(out: *std.Io.Writer, err: errors.ApiError) std.Io.Writer.Error!void {
    try out.writeAll("data: ");
    // The body ends its line; the blank line ends the event.
    try errors.write(out, err);
    try out.writeAll("\n");
}

// ----- tests -----

const testing = std.testing;

fn expectRefused(body: []const u8, code: []const u8, param: ?[]const u8) !void {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    var problem: Problem = .{};
    testing.expectError(error.Refused, parse(arena_state.allocator(), body, &problem)) catch |err| {
        std.debug.print("accepted: {s}\n", .{body});
        return err;
    };
    errdefer std.debug.print("{s}: {s} {?s} ({s})\n", .{ body, problem.code, problem.param, problem.message });
    try testing.expectEqualStrings(code, problem.code);
    if (param) |p| try testing.expectEqualStrings(p, problem.param.?) else try testing.expect(problem.param == null);
}

test "the speech request: OpenAI's fields and nuclis's, defaults where absent" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var problem: Problem = .{};
    const full = try parse(arena,
        \\{"model": "qwen3-tts-1.7b-customvoice", "input": "Hello there.", "voice": "vivian",
        \\ "instructions": "Whisper.", "response_format": "pcm", "stream_format": "sse",
        \\ "speed": 1.0, "seed": 7, "language": "english", "user": "ignored"}
    , &problem);
    try testing.expectEqualStrings("qwen3-tts-1.7b-customvoice", full.model.?);
    try testing.expectEqualStrings("Hello there.", full.input);
    try testing.expectEqualStrings("vivian", full.voice.?);
    try testing.expectEqualStrings("Whisper.", full.instructions.?);
    try testing.expectEqual(Format.pcm, full.format);
    try testing.expectEqual(StreamFormat.sse, full.stream);
    try testing.expectEqual(@as(?u64, 7), full.seed);
    try testing.expectEqualStrings("english", full.language.?);
    const bare = try parse(arena, "{\"input\": \"Hi.\", \"instructions\": \"  \", \"speed\": 1}", &problem);
    try testing.expectEqual(Format.wav, bare.format);
    try testing.expectEqual(StreamFormat.audio, bare.stream);
    try testing.expect(bare.model == null and bare.voice == null and bare.instructions == null and bare.seed == null);
    try testing.expectEqualStrings("audio/wav", contentType(bare));
    try testing.expectEqualStrings("text/event-stream", contentType(full));
    try testing.expectEqualStrings("audio/pcm", contentType(.{ .input = "x", .format = .pcm }));
}

test "the speech request: every refusal names its code and field" {
    try expectRefused("{\"input\":", "invalid_json", null);
    try expectRefused("[1]", "invalid_request", "body");
    try expectRefused("{}", "invalid_request", "input");
    try expectRefused("{\"input\": 3}", "invalid_request", "input");
    try expectRefused("{\"input\": \"  \"}", "invalid_request", "input");
    try expectRefused("{\"input\": \"x\", \"response_format\": \"mp3\"}", "unsupported_feature", "response_format");
    try expectRefused("{\"input\": \"x\", \"response_format\": \"opus\"}", "unsupported_feature", "response_format");
    try expectRefused("{\"input\": \"x\", \"response_format\": \"ogg\"}", "invalid_request", "response_format");
    try expectRefused("{\"input\": \"x\", \"stream_format\": \"chunks\"}", "invalid_request", "stream_format");
    try expectRefused("{\"input\": \"x\", \"speed\": 1.25}", "unsupported_feature", "speed");
    try expectRefused("{\"input\": \"x\", \"speed\": \"fast\"}", "invalid_request", "speed");
    try expectRefused("{\"input\": \"x\", \"voice\": {\"name\": \"v1\"}}", "invalid_request", "voice");
    var problem: Problem = .{};
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const by_id = try parse(arena_state.allocator(), "{\"input\": \"x\", \"voice\": {\"id\": \"anna\"}}", &problem);
    try testing.expectEqualStrings("anna", by_id.voice.?);
    try testing.expect(by_id.voice_by_id);
    try expectRefused("{\"input\": \"x\", \"voice\": 1}", "invalid_request", "voice");
    try expectRefused("{\"input\": \"x\", \"seed\": -1}", "invalid_request", "seed");
    try expectRefused("{\"input\": \"x\", \"model\": 5}", "invalid_request", "model");
    try expectRefused("{\"input\": \"x\", \"instructions\": []}", "invalid_request", "instructions");
    try expectRefused("{\"input\": \"\xff\"}", "invalid_json", null);
    const long = try testing.allocator.alloc(u8, max_input_bytes + 16);
    defer testing.allocator.free(long);
    @memset(long, 'a');
    @memcpy(long[0..11], "{\"input\": \"");
    @memcpy(long[long.len - 2 ..], "\"}");
    try expectRefused(long, "request_too_large", "input");
}

test "the stream's bytes: PCM, a header for an unknown length, and the events" {
    const h = streamingWavHeader();
    try testing.expectEqualStrings("RIFF", h[0..4]);
    try testing.expectEqualStrings("WAVEfmt ", h[8..16]);
    try testing.expectEqual(@as(u32, 24000), std.mem.readInt(u32, h[24..28], .little));
    try testing.expectEqual(std.math.maxInt(u32), std.mem.readInt(u32, h[40..44], .little));
    var bytes: [6]u8 = undefined;
    const pcm = pcm16(&.{ 1.5, -1, 0 }, &bytes);
    try testing.expectEqual(@as(i16, 32767), std.mem.readInt(i16, pcm[0..2], .little));
    try testing.expectEqual(@as(i16, -32767), std.mem.readInt(i16, pcm[2..4], .little));
    try testing.expectEqual(@as(i16, 0), std.mem.readInt(i16, pcm[4..6], .little));

    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    try writeDelta(&out.writer, "\x01\x02\x03\x04");
    try writeDone(&out.writer, .{ .input_tokens = 12, .output_tokens = 40 }, "length");
    var err: errors.ApiError = .init(.internal_server_error, "model_failed", "the device failed");
    err.param = null;
    try writeStreamError(&out.writer, err);
    // Each event is one `data:` line of JSON, then a blank line.
    var events = std.mem.splitSequence(u8, out.written(), "\n\n");
    var parsed: [3]std.json.Parsed(Value) = undefined;
    var n: usize = 0;
    while (events.next()) |event| {
        if (event.len == 0) continue;
        try testing.expect(std.mem.startsWith(u8, event, "data: ") and std.mem.indexOfScalar(u8, event, '\n') == null);
        parsed[n] = try std.json.parseFromSlice(Value, testing.allocator, event["data: ".len..], .{});
        n += 1;
    }
    defer for (parsed[0..n]) |p| p.deinit();
    try testing.expectEqual(@as(usize, 3), n);
    try testing.expectEqualStrings("speech.audio.delta", parsed[0].value.object.get("type").?.string);
    try testing.expectEqualStrings("AQIDBA==", parsed[0].value.object.get("audio").?.string);
    try testing.expectEqualStrings("speech.audio.done", parsed[1].value.object.get("type").?.string);
    try testing.expectEqual(@as(i64, 52), parsed[1].value.object.get("usage").?.object.get("total_tokens").?.integer);
    try testing.expectEqualStrings("length", parsed[1].value.object.get("finish_reason").?.string);
    try testing.expectEqualStrings("model_failed", parsed[2].value.object.get("error").?.object.get("code").?.string);
    // A chunk's room covers its frames, the header, and the event's framing.
    try testing.expect(maxChunkBytes(4, 1920, .sse) > std.base64.standard.Encoder.calcSize(4 * 1920 * 2 + 44));
    try testing.expectEqual(@as(usize, 4 * 1920 * 2 + 44), maxChunkBytes(4, 1920, .audio));
}
