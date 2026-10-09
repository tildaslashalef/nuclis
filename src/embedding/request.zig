//! The embedding request: inputs, each an ordered list of parts that gives
//! one vector, and the options every input shares (`task` and `title`, which
//! render Google's prefix at the front of an input; the width; truncation;
//! the image budget). Shared
//! by `nuclis embed` and the API; knows no HTTP. Text is embedded exactly as
//! given unless a task is named (docs/models/embeddinggemma.md § Tokenizer
//! and task prefixes).
const std = @import("std");
const inference = @import("inference");
const config = @import("../config.zig");
const data_url = @import("../data_url.zig");

const embed = inference.embed;

/// Inputs one request may carry (OpenAI's limit for its embeddings call).
pub const max_inputs = 2048;
/// The most bytes of one input's text, well above what 8192 tokens spell.
pub const max_text_bytes = 1024 * 1024;
/// The most bytes of one encoded image (the decoder's own bound).
pub const max_image_bytes = inference.vision.image.max_bytes;

/// What the vectors are for. A query task puts `task: … | query: ` in front
/// of an input's text; `document` puts `title: … | text: `.
pub const Task = enum {
    search_query,
    document,
    question_answering,
    fact_checking,
    code_retrieval,
    classification,
    clustering,
    similarity,

    /// The text that opens an input. `title` is a document's (none renders
    /// `none`); other tasks take none.
    pub fn prefix(self: Task, arena: std.mem.Allocator, title: ?[]const u8) ![]const u8 {
        const query = switch (self) {
            .document => return arena.print("title: {s} | text: ", .{title orelse "none"}),
            .search_query => "search result",
            .question_answering => "question answering",
            .fact_checking => "fact checking",
            .code_retrieval => "code retrieval",
            .classification => "classification",
            .clustering => "clustering",
            .similarity => "sentence similarity",
        };
        return arena.print("task: {s} | query: ", .{query});
    }
};

pub const Part = union(enum) {
    text: []const u8,
    /// Encoded image bytes.
    image: []const u8,
    /// Encoded audio bytes.
    audio: []const u8,
};

pub const Input = struct {
    parts: []const Part,
    /// How reports name it: `input[3]`, a file name, a line of a file.
    label: []const u8,
};

pub const Request = struct {
    inputs: []const Input,
    task: ?Task = null,
    title: ?[]const u8 = null,
    /// One of `embed.widths`.
    dimensions: usize = embed.dimensions,
    /// Cut an input over the token limit instead of refusing it; the
    /// response says which were cut.
    truncate: bool = false,
    /// Soft tokens per image, one of `embed.image_budgets`: it changes the
    /// vector, so the response records it.
    image_tokens: u32 = embed.default_image_tokens,

    pub fn hasImages(self: Request) bool {
        for (self.inputs) |input| for (input.parts) |part| if (part == .image) return true;
        return false;
    }
};

/// The budgets an image may take, as the messages spell them.
pub const image_budgets_text = "70, 140, 280, 560, or 1120";

pub const Error = error{ InvalidRequest, RequestTooLarge, UnsupportedDimensions, UnsupportedModality };

/// The request's own consistency: inputs present and bounded, a trained
/// width, a title only for documents, no empty input.
pub fn check(request: Request, diag: *config.Diagnostic) Error!void {
    if (request.inputs.len == 0) {
        diag.set("nothing to embed: give at least one input", .{});
        return error.InvalidRequest;
    }
    if (request.inputs.len > max_inputs) {
        diag.set("at most {d} inputs per request; this one has {d}", .{ max_inputs, request.inputs.len });
        return error.RequestTooLarge;
    }
    if (std.mem.indexOfScalar(usize, &embed.widths, request.dimensions) == null) {
        diag.set("dimensions must be 768, 512, 256, or 128 (the widths the model is trained for), not {d}", .{request.dimensions});
        return error.UnsupportedDimensions;
    }
    if (request.title != null and request.task != .document) {
        diag.set("a title belongs to a document: give task document with it", .{});
        return error.InvalidRequest;
    }
    if (std.mem.indexOfScalar(u32, &embed.image_budgets, request.image_tokens) == null) {
        diag.set("image tokens are {s} (the budgets the model's processor takes), not {d}", .{ image_budgets_text, request.image_tokens });
        return error.InvalidRequest;
    }
    for (request.inputs) |input| {
        if (input.parts.len == 0) {
            diag.set("{s} has no parts", .{input.label});
            return error.InvalidRequest;
        }
        for (input.parts) |part| switch (part) {
            .text => |t| if (t.len > max_text_bytes) {
                diag.set("{s}: {d} bytes of text, at most {d}", .{ input.label, t.len, max_text_bytes });
                return error.RequestTooLarge;
            },
            .image => |bytes| if (bytes.len > max_image_bytes) {
                diag.set("{s}: an image of {d} bytes, at most {d}", .{ input.label, bytes.len, max_image_bytes });
                return error.RequestTooLarge;
            },
            .audio => {},
        };
    }
}

/// The input as the engine reads it: the task's prefix opens it, before
/// any media part (as sentence-transformers places a prompt, so an image
/// alone gets one too), and the engine joins it to a text part that
/// follows. Audio parts are refused until their encoder exists.
pub fn render(arena: std.mem.Allocator, request: Request, input: Input, diag: *config.Diagnostic) !embed.Input {
    const lead: usize = @intFromBool(request.task != null);
    const parts = try arena.alloc(embed.Part, input.parts.len + lead);
    if (request.task) |task| parts[0] = .{ .text = try task.prefix(arena, request.title) };
    for (input.parts, parts[lead..]) |part, *out| out.* = switch (part) {
        .text => |t| .{ .text = t },
        .image => |bytes| .{ .image = bytes },
        .audio => {
            diag.set("{s}: audio parts are not supported yet; nuclis embeds text and images", .{input.label});
            return error.UnsupportedModality;
        },
    };
    return .{ .parts = parts };
}

/// One input of a JSON request or a line of an inputs file: a string, or a
/// list of chat content parts (`{"type":"text","text":…}`, `image_url` with
/// a base64 data URL, `input_audio`), in order.
pub fn inputFromJson(arena: std.mem.Allocator, value: std.json.Value, label: []const u8, diag: *config.Diagnostic) !Input {
    switch (value) {
        .string => |s| {
            const parts = try arena.alloc(Part, 1);
            parts[0] = .{ .text = s };
            return .{ .parts = parts, .label = label };
        },
        .array => |a| {
            const parts = try arena.alloc(Part, a.items.len);
            for (a.items, parts, 0..) |item, *part, i| {
                const object = if (item == .object) item.object else {
                    diag.set("{s}: part {d} is not an object ({{\"type\": \"text\", \"text\": …}})", .{ label, i });
                    return error.InvalidRequest;
                };
                const kind = if (object.get("type")) |t| if (t == .string) t.string else "" else "";
                if (std.mem.eql(u8, kind, "text")) {
                    const text = object.get("text") orelse .null;
                    if (text != .string) {
                        diag.set("{s}: part {d} is a text part without a \"text\" string", .{ label, i });
                        return error.InvalidRequest;
                    }
                    part.* = .{ .text = text.string };
                } else if (std.mem.eql(u8, kind, "image_url")) {
                    part.* = .{ .image = try imageFromJson(arena, object.get("image_url") orelse .null, label, i, diag) };
                } else if (std.mem.eql(u8, kind, "input_audio")) {
                    diag.set("{s}: part {d} is input_audio; nuclis embeds text and image parts for now", .{ label, i });
                    return error.UnsupportedModality;
                } else {
                    diag.set("{s}: part {d} has type \"{s}\"; a part is text, image_url, or input_audio", .{ label, i, kind });
                    return error.InvalidRequest;
                }
            }
            return .{ .parts = parts, .label = label };
        },
        else => {
            diag.set("{s}: an input is a string or a list of content parts", .{label});
            return error.InvalidRequest;
        },
    }
}

/// The bytes of an `image_url` part's value: `{"url": …}` or the URL itself,
/// a base64 data URL (nuclis fetches no URL) or bare base64.
fn imageFromJson(arena: std.mem.Allocator, value: std.json.Value, label: []const u8, i: usize, diag: *config.Diagnostic) ![]const u8 {
    const url = switch (value) {
        .string => |s| s,
        .object => |o| if (o.get("url")) |u| if (u == .string) u.string else null else null,
        else => null,
    } orelse {
        diag.set("{s}: part {d} is an image_url part without a \"url\"", .{ label, i });
        return error.InvalidRequest;
    };
    if (std.mem.startsWith(u8, url, "http://") or std.mem.startsWith(u8, url, "https://")) {
        diag.set("{s}: part {d}: nuclis fetches no URL; send the image as a base64 data URL", .{ label, i });
        return error.UnsupportedModality;
    }
    const encoded = data_url.payload(url) orelse {
        diag.set("{s}: part {d}: a data URL must be base64 (data:<type>;base64,…)", .{ label, i });
        return error.InvalidRequest;
    };
    return data_url.decode(arena, encoded, max_image_bytes) catch |err| switch (err) {
        error.NotBase64 => {
            diag.set("{s}: part {d}: the image is not base64", .{ label, i });
            return error.InvalidRequest;
        },
        error.TooLarge => {
            diag.set("{s}: part {d}: an image is at most {d} bytes", .{ label, i, max_image_bytes });
            return error.RequestTooLarge;
        },
        error.OutOfMemory => return error.OutOfMemory,
    };
}

/// An inputs file: one JSON input per non-blank line (`inputFromJson`),
/// labelled `<name>:<line>`.
pub fn inputsFromJsonLines(arena: std.mem.Allocator, bytes: []const u8, name: []const u8, out: *std.ArrayList(Input), diag: *config.Diagnostic) !void {
    var lines = std.mem.splitScalar(u8, bytes, '\n');
    var number: usize = 0;
    while (lines.next()) |raw| {
        number += 1;
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0) continue;
        if (out.items.len == max_inputs) {
            diag.set("{s}: more than {d} inputs; split the file", .{ name, max_inputs });
            return error.RequestTooLarge;
        }
        const label = try arena.print("{s}:{d}", .{ name, number });
        const value = std.json.parseFromSliceLeaky(std.json.Value, arena, line, .{}) catch |err| {
            diag.set("{s}: not a JSON value ({s}); a line is a string or a list of content parts", .{ label, @errorName(err) });
            return error.InvalidRequest;
        };
        try out.append(arena, try inputFromJson(arena, value, label, diag));
    }
}

/// Why an API request was refused: a 400 with this code, message, and
/// field.
pub const Problem = struct {
    code: []const u8 = "invalid_request",
    message: []const u8 = "",
    param: ?[]const u8 = null,
};

/// How the API writes a vector: JSON numbers, or base64 of its
/// little-endian f32 bytes (what OpenAI's Python SDK asks for by default).
pub const Encoding = enum { float, base64 };

/// An OpenAI embeddings request with nuclis's extensions.
pub const Parsed = struct {
    request: Request,
    model: ?[]const u8 = null,
    encoding: Encoding = .float,
};

/// Reads OpenAI's embeddings body: `model`, `input` (a string, or a list
/// whose items are strings or lists of content parts), `dimensions`,
/// `encoding_format`, `user` (ignored), and the extensions `task`, `title`,
/// `truncate` and `image_tokens`. Token arrays are refused: a client that sends them
/// tokenized with another model's vocabulary. `Refused` sets `problem`.
pub fn fromJson(arena: std.mem.Allocator, root: std.json.Value, problem: *Problem) error{ Refused, OutOfMemory }!Parsed {
    const R = struct {
        fn refuse(arena_: std.mem.Allocator, p: *Problem, code: []const u8, param: ?[]const u8, comptime format: []const u8, args: anytype) error{ Refused, OutOfMemory } {
            p.* = .{ .code = code, .param = param, .message = try arena_.print(format, args) };
            return error.Refused;
        }
    };
    if (root != .object) return R.refuse(arena, problem, "invalid_request", "body", "the request body must be a JSON object", .{});
    const o = root.object;
    var parsed: Parsed = .{ .request = .{ .inputs = &.{} } };
    if (o.get("model")) |m| switch (m) {
        .string => |name| parsed.model = name,
        .null => {},
        else => return R.refuse(arena, problem, "invalid_request", "model", "\"model\" must be a string", .{}),
    };
    const token_hint = "token arrays are refused: they are another model's tokens (LangChain's OpenAIEmbeddings sends them unless check_embedding_ctx_length=False); send text";
    const input_value = o.get("input") orelse return R.refuse(arena, problem, "invalid_request", "input", "\"input\" is required", .{});
    const items: []const std.json.Value = switch (input_value) {
        .string => (&input_value)[0..1],
        .array => |a| a.items,
        else => return R.refuse(arena, problem, "invalid_request", "input", "\"input\" is a string or a list", .{}),
    };
    if (items.len == 0) return R.refuse(arena, problem, "invalid_request", "input", "\"input\" is empty", .{});
    if (items.len > max_inputs) return R.refuse(arena, problem, "request_too_large", "input", "at most {d} inputs per request; this one has {d}", .{ max_inputs, items.len });
    const inputs = try arena.alloc(Input, items.len);
    for (items, inputs, 0..) |item, *input, i| {
        const label = try arena.print("input[{d}]", .{i});
        switch (item) {
            .integer => return R.refuse(arena, problem, "unsupported_feature", "input", "{s}", .{token_hint}),
            .array => |a| if (a.items.len > 0 and a.items[0] == .integer) return R.refuse(arena, problem, "unsupported_feature", "input", "{s}", .{token_hint}),
            .string => {},
            else => return R.refuse(arena, problem, "invalid_request", "input", "{s}: an input is a string or a list of content parts", .{label}),
        }
        var diag: config.Diagnostic = .{};
        input.* = inputFromJson(arena, item, label, &diag) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.UnsupportedModality => return R.refuse(arena, problem, "unsupported_feature", "input", "{s}", .{diag.message()}),
            error.RequestTooLarge => return R.refuse(arena, problem, "request_too_large", "input", "{s}", .{diag.message()}),
            else => return R.refuse(arena, problem, "invalid_request", "input", "{s}", .{diag.message()}),
        };
    }
    parsed.request.inputs = inputs;
    if (o.get("dimensions")) |d| switch (d) {
        .null => {},
        .integer => |n| {
            if (n < 0 or std.mem.indexOfScalar(usize, &embed.widths, @intCast(n)) == null)
                return R.refuse(arena, problem, "unsupported_feature", "dimensions", "dimensions are 768, 512, 256, or 128 (the widths the model is trained for), not {d}", .{n});
            parsed.request.dimensions = @intCast(n);
        },
        else => return R.refuse(arena, problem, "invalid_request", "dimensions", "\"dimensions\" must be a whole number", .{}),
    };
    if (o.get("encoding_format")) |f| switch (f) {
        .null => {},
        .string => |name| parsed.encoding = std.meta.stringToEnum(Encoding, name) orelse
            return R.refuse(arena, problem, "invalid_request", "encoding_format", "\"encoding_format\" is \"float\" or \"base64\", not \"{s}\"", .{name}),
        else => return R.refuse(arena, problem, "invalid_request", "encoding_format", "\"encoding_format\" must be a string", .{}),
    };
    if (o.get("task")) |t| switch (t) {
        .null => {},
        .string => |name| parsed.request.task = std.meta.stringToEnum(Task, name) orelse
            return R.refuse(arena, problem, "invalid_request", "task", "\"task\" is search_query, document, question_answering, fact_checking, code_retrieval, classification, clustering, or similarity, not \"{s}\"", .{name}),
        else => return R.refuse(arena, problem, "invalid_request", "task", "\"task\" must be a string", .{}),
    };
    if (o.get("title")) |t| switch (t) {
        .null => {},
        .string => |title| parsed.request.title = title,
        else => return R.refuse(arena, problem, "invalid_request", "title", "\"title\" must be a string", .{}),
    };
    if (parsed.request.title != null and parsed.request.task != .document)
        return R.refuse(arena, problem, "invalid_request", "title", "a title belongs to a document: send \"task\": \"document\" with it", .{});
    if (o.get("truncate")) |t| switch (t) {
        .null => {},
        .bool => |b| parsed.request.truncate = b,
        else => return R.refuse(arena, problem, "invalid_request", "truncate", "\"truncate\" must be true or false", .{}),
    };
    if (o.get("image_tokens")) |t| switch (t) {
        .null => {},
        .integer => |n| {
            if (n < 0 or n > std.math.maxInt(u32) or std.mem.indexOfScalar(u32, &embed.image_budgets, @intCast(n)) == null)
                return R.refuse(arena, problem, "invalid_request", "image_tokens", "\"image_tokens\" is {s}, not {d}", .{ image_budgets_text, n });
            parsed.request.image_tokens = @intCast(n);
        },
        else => return R.refuse(arena, problem, "invalid_request", "image_tokens", "\"image_tokens\" must be a whole number", .{}),
    };
    var diag: config.Diagnostic = .{};
    check(parsed.request, &diag) catch |err| return R.refuse(arena, problem, if (err == error.RequestTooLarge) "request_too_large" else "invalid_request", "input", "{s}", .{diag.message()});
    return parsed;
}

const testing = std.testing;

test "an OpenAI body: inputs in every accepted shape, the options, and each refusal with its field" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var problem: Problem = .{};
    const body =
        \\{"model":"e","input":["a",[{"type":"text","text":"b"},{"type":"text","text":"c"}]],"dimensions":256,
        \\ "encoding_format":"base64","user":"u","task":"document","title":"T","truncate":true}
    ;
    const parsed = try fromJson(arena, try std.json.parseFromSliceLeaky(std.json.Value, arena, body, .{}), &problem);
    try testing.expectEqualStrings("e", parsed.model.?);
    try testing.expectEqual(@as(usize, 2), parsed.request.inputs.len);
    try testing.expectEqualStrings("c", parsed.request.inputs[1].parts[1].text);
    try testing.expectEqualStrings("input[1]", parsed.request.inputs[1].label);
    try testing.expectEqual(Encoding.base64, parsed.encoding);
    try testing.expectEqual(@as(usize, 256), parsed.request.dimensions);
    try testing.expect(parsed.request.truncate and parsed.request.task.? == .document);
    const one = try fromJson(arena, try std.json.parseFromSliceLeaky(std.json.Value, arena, "{\"input\":\"x\"}", .{}), &problem);
    try testing.expectEqual(@as(usize, 1), one.request.inputs.len);
    try testing.expectEqual(@as(usize, 768), one.request.dimensions);
    try testing.expectEqual(embed.default_image_tokens, one.request.image_tokens);
    const image_body = "{\"input\":[[{\"type\":\"text\",\"text\":\"a \"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,aGk=\"}}]],\"image_tokens\":70}";
    const image = try fromJson(arena, try std.json.parseFromSliceLeaky(std.json.Value, arena, image_body, .{}), &problem);
    try testing.expectEqualStrings("hi", image.request.inputs[0].parts[1].image);
    try testing.expectEqual(@as(u32, 70), image.request.image_tokens);
    try testing.expect(image.request.hasImages() and !one.request.hasImages());

    const cases = [_]struct { body: []const u8, code: []const u8, param: []const u8 }{
        .{ .body = "[]", .code = "invalid_request", .param = "body" },
        .{ .body = "{}", .code = "invalid_request", .param = "input" },
        .{ .body = "{\"input\":[]}", .code = "invalid_request", .param = "input" },
        .{ .body = "{\"input\":[1,2,3]}", .code = "unsupported_feature", .param = "input" },
        .{ .body = "{\"input\":[[1,2],[3]]}", .code = "unsupported_feature", .param = "input" },
        .{ .body = "{\"input\":[{\"type\":\"text\",\"text\":\"x\"}]}", .code = "invalid_request", .param = "input" },
        .{ .body = "{\"input\":[[{\"type\":\"image_url\",\"image_url\":{\"url\":\"https://x/y.png\"}}]]}", .code = "unsupported_feature", .param = "input" },
        .{ .body = "{\"input\":[[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png,raw\"}}]]}", .code = "invalid_request", .param = "input" },
        .{ .body = "{\"input\":[[{\"type\":\"input_audio\",\"input_audio\":{}}]]}", .code = "unsupported_feature", .param = "input" },
        .{ .body = "{\"input\":\"x\",\"image_tokens\":300}", .code = "invalid_request", .param = "image_tokens" },
        .{ .body = "{\"input\":\"x\",\"image_tokens\":\"280\"}", .code = "invalid_request", .param = "image_tokens" },
        .{ .body = "{\"input\":\"x\",\"dimensions\":300}", .code = "unsupported_feature", .param = "dimensions" },
        .{ .body = "{\"input\":\"x\",\"dimensions\":\"768\"}", .code = "invalid_request", .param = "dimensions" },
        .{ .body = "{\"input\":\"x\",\"encoding_format\":\"int8\"}", .code = "invalid_request", .param = "encoding_format" },
        .{ .body = "{\"input\":\"x\",\"task\":\"query\"}", .code = "invalid_request", .param = "task" },
        .{ .body = "{\"input\":\"x\",\"title\":\"t\"}", .code = "invalid_request", .param = "title" },
        .{ .body = "{\"input\":\"x\",\"truncate\":1}", .code = "invalid_request", .param = "truncate" },
        .{ .body = "{\"input\":\"x\",\"model\":3}", .code = "invalid_request", .param = "model" },
    };
    for (cases) |case| {
        errdefer std.debug.print("case: {s}\n", .{case.body});
        try testing.expectError(error.Refused, fromJson(arena, try std.json.parseFromSliceLeaky(std.json.Value, arena, case.body, .{}), &problem));
        try testing.expectEqualStrings(case.code, problem.code);
        try testing.expectEqualStrings(case.param, problem.param.?);
    }
}

test "prefixes are Google's, and only a document takes a title" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try testing.expectEqualStrings("task: search result | query: ", try Task.search_query.prefix(arena, null));
    try testing.expectEqualStrings("task: sentence similarity | query: ", try Task.similarity.prefix(arena, null));
    try testing.expectEqualStrings("task: code retrieval | query: ", try Task.code_retrieval.prefix(arena, null));
    try testing.expectEqualStrings("title: none | text: ", try Task.document.prefix(arena, null));
    try testing.expectEqualStrings("title: Aurora | text: ", try Task.document.prefix(arena, "Aurora"));

    var diag: config.Diagnostic = .{};
    const one = [_]Input{.{ .parts = &.{.{ .text = "x" }}, .label = "input[0]" }};
    try check(.{ .inputs = &one }, &diag);
    try testing.expectError(error.InvalidRequest, check(.{ .inputs = &one, .task = .search_query, .title = "t" }, &diag));
    try testing.expectError(error.UnsupportedDimensions, check(.{ .inputs = &one, .dimensions = 300 }, &diag));
    try testing.expectError(error.InvalidRequest, check(.{ .inputs = &.{} }, &diag));
    try testing.expectError(error.InvalidRequest, check(.{ .inputs = &.{.{ .parts = &.{}, .label = "input[0]" }} }, &diag));
}

test "the prefix opens the input, before any media; no task leaves the parts as given" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diag: config.Diagnostic = .{};
    const input: Input = .{ .parts = &.{ .{ .text = "a" }, .{ .text = "b" } }, .label = "input[0]" };
    const plain = try render(arena, .{ .inputs = &.{input} }, input, &diag);
    try testing.expectEqualStrings("a", plain.parts[0].text);
    const query = try render(arena, .{ .inputs = &.{input}, .task = .search_query }, input, &diag);
    try testing.expectEqualStrings("task: search result | query: ", query.parts[0].text);
    try testing.expectEqualStrings("a", query.parts[1].text);
    try testing.expectEqualStrings("b", query.parts[2].text);
    const image: Input = .{ .parts = &.{ .{ .image = "PNG" }, .{ .text = "c" } }, .label = "input[1]" };
    const rendered = try render(arena, .{ .inputs = &.{image}, .task = .document, .title = "T" }, image, &diag);
    try testing.expectEqualStrings("title: T | text: ", rendered.parts[0].text);
    try testing.expectEqualStrings("PNG", rendered.parts[1].image);
    const audio: Input = .{ .parts = &.{.{ .audio = "RIFF" }}, .label = "input[2]" };
    try testing.expectError(error.UnsupportedModality, render(arena, .{ .inputs = &.{audio} }, audio, &diag));
}

test "an inputs file: strings and part lists per line, blanks skipped, errors name the line" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diag: config.Diagnostic = .{};
    var inputs: std.ArrayList(Input) = .empty;
    try inputsFromJsonLines(arena, "\"one\"\n\n[{\"type\":\"text\",\"text\":\"two\"},{\"type\":\"text\",\"text\":\" more\"}]\r\n", "in.jsonl", &inputs, &diag);
    try testing.expectEqual(@as(usize, 2), inputs.items.len);
    try testing.expectEqualStrings("one", inputs.items[0].parts[0].text);
    try testing.expectEqualStrings("in.jsonl:3", inputs.items[1].label);
    try testing.expectEqualStrings(" more", inputs.items[1].parts[1].text);
    inputs.clearRetainingCapacity();
    try testing.expectError(error.InvalidRequest, inputsFromJsonLines(arena, "\"ok\"\nnot json\n", "in.jsonl", &inputs, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "in.jsonl:2") != null);
    try testing.expectError(error.InvalidRequest, inputsFromJsonLines(arena, "[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:\"}}]", "f", &inputs, &diag));
    try inputsFromJsonLines(arena, "[{\"type\":\"image_url\",\"image_url\":\"aGk=\"}]", "f", &inputs, &diag);
    try testing.expectEqualStrings("hi", inputs.items[inputs.items.len - 1].parts[0].image);
    try testing.expectError(error.InvalidRequest, inputsFromJsonLines(arena, "[{\"type\":\"video\"}]", "f", &inputs, &diag));
    try testing.expectError(error.InvalidRequest, inputsFromJsonLines(arena, "42", "f", &inputs, &diag));
}
