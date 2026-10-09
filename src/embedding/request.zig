//! The embedding request: inputs, each an ordered list of parts that gives
//! one vector, and the options every input shares (`task` and `title`, which
//! render Google's prefixes onto text parts; the width; truncation). Shared
//! by `nuclis embed` and the API; knows no HTTP. Text is embedded exactly as
//! given unless a task is named (docs/models/embeddinggemma.md § Tokenizer
//! and task prefixes).
const std = @import("std");
const inference = @import("inference");
const config = @import("../config.zig");

const embed = inference.embed;

/// Inputs one request may carry (OpenAI's limit for its embeddings call).
pub const max_inputs = 2048;
/// The most bytes of one input's text, well above what 8192 tokens spell.
pub const max_text_bytes = 1024 * 1024;

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

    /// The text in front of an input's first text part. `title` is a
    /// document's (none renders `none`); other tasks take none.
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
};

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
            else => {},
        };
    }
}

/// The input as the engine reads it: the task's prefix in front of its first
/// text part (an input without text gets none). Image and audio parts are
/// refused until their encoders exist.
pub fn render(arena: std.mem.Allocator, request: Request, input: Input, diag: *config.Diagnostic) !embed.Input {
    const parts = try arena.alloc(embed.Part, input.parts.len);
    var prefixed = request.task == null;
    for (input.parts, parts) |part, *out| switch (part) {
        .text => |t| {
            out.* = .{ .text = if (prefixed) t else try std.mem.concat(arena, u8, &.{ try request.task.?.prefix(arena, request.title), t }) };
            prefixed = true;
        },
        .image, .audio => {
            diag.set("{s}: {s} parts are not supported yet; nuclis embeds text", .{ input.label, @tagName(part) });
            return error.UnsupportedModality;
        },
    };
    return .{ .parts = parts };
}

/// One input of a JSON request or a line of an inputs file: a string, or a
/// list of chat content parts (`{"type":"text","text":…}`, `image_url`,
/// `input_audio`), in order.
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
                } else if (std.mem.eql(u8, kind, "image_url") or std.mem.eql(u8, kind, "input_audio")) {
                    diag.set("{s}: part {d} is {s}; nuclis embeds text parts only for now", .{ label, i, kind });
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

const testing = std.testing;

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

test "the prefix goes in front of the first text part only; no task leaves text as given" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diag: config.Diagnostic = .{};
    const input: Input = .{ .parts = &.{ .{ .text = "a" }, .{ .text = "b" } }, .label = "input[0]" };
    const plain = try render(arena, .{ .inputs = &.{input} }, input, &diag);
    try testing.expectEqualStrings("a", plain.parts[0].text);
    const query = try render(arena, .{ .inputs = &.{input}, .task = .search_query }, input, &diag);
    try testing.expectEqualStrings("task: search result | query: a", query.parts[0].text);
    try testing.expectEqualStrings("b", query.parts[1].text);
    const image: Input = .{ .parts = &.{.{ .image = "PNG" }}, .label = "input[1]" };
    try testing.expectError(error.UnsupportedModality, render(arena, .{ .inputs = &.{image} }, image, &diag));
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
    try testing.expectError(error.UnsupportedModality, inputsFromJsonLines(arena, "[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:\"}}]", "f", &inputs, &diag));
    try testing.expectError(error.InvalidRequest, inputsFromJsonLines(arena, "[{\"type\":\"video\"}]", "f", &inputs, &diag));
    try testing.expectError(error.InvalidRequest, inputsFromJsonLines(arena, "42", "f", &inputs, &diag));
}
