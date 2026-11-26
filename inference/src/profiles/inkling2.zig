//! Bounded prompt profile for Inkling 2, implemented from the chat template
//! embedded in its GGUF (docs/models/inkling2.md § Prompt profile). A turn
//! is a run of messages `<role><content-type>…<|end_message|>`; the model's
//! turn ends with `<|content_model_end_sampling|>`, the stop token. There is
//! no generation prompt: the model writes `<|message_model|>` before every
//! segment itself, and the content-type token routes it (the stream decoder's
//! channel grammar). The effort line is a system message before the first
//! other message, so it is part of `prefix`.
//!
//! Tools are declared as one JSON array (keys sorted, compact) in a
//! `tool_declare` system message; a call is
//! `<|message_model|>NAME<|content_invoke_tool_json|>{"name":…,"args":{…}}`
//! with the arguments' top-level keys sorted; a result is a `<|message_tool|>NAME`
//! message. Fixtures under `fixtures/inkling2-*.json` are written from the
//! template's text. It owns no tokenizer or model equations.
const std = @import("std");
const alloc_check = @import("../alloc_check.zig");
const sampling = @import("../sampling/root.zig");

/// Exact GGUF template this profile implements (`tokenizer.chat_template`,
/// 6,868 bytes); `profiles.forDocument` selects the profile by this digest.
pub const template_sha256 = "53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc";
pub const template_aliases = [_][]const u8{};

const profiles = @import("root.zig");
pub const Role = profiles.Role;
pub const Effort = profiles.Effort;
pub const Message = profiles.Message;
pub const Limits = profiles.Limits;
pub const Error = profiles.Error;
const Builder = profiles.Builder;
pub const SamplingOverrides = profiles.SamplingOverrides;

const user = "<|message_user|>";
const model = "<|message_model|>";
const system = "<|message_system|>";
const tool_role = "<|message_tool|>";
const text_part = "<|content_text|>";
const thinking_part = "<|content_thinking|>";
const call_part = "<|content_invoke_tool_json|>";
const xml_part = "<|content_xml|>";
const image_part = "<|content_image|>";
const end = "<|end_message|>";
const end_sampling = "<|content_model_end_sampling|>";

/// The file's EOS and EOT (200006).
pub const stop_tokens = [_][]const u8{end_sampling};
/// One per merged image row, inside a `<|content_image|>` user message.
pub const image_placeholder: ?[]const u8 = "<|unused_200054|>";
/// Every level renders distinctly (the template's effort table).
pub const efforts = [_]Effort{ .off, .low, .medium, .high, .xhigh };
pub const reasoning: profiles.Reasoning = .{ .open = model ++ thinking_part, .close = end };
/// None: the render ends with the last message.
pub const generation_prompts = [_][]const u8{};
/// History renders reasoning and answer as the segments the model wrote.
pub fn rewritesTurn(_: Effort) bool {
    return false;
}
pub const stream_markers: profiles.StreamMarkers = .{
    .open = model,
    .close = end,
    .tool_parse = parseTool,
    .channel = .{ .start = model, .message = text_part, .eom = end, .thinking = thinking_part, .tool = call_part },
};
/// The file adds no BOS and the template writes none.
pub const bos_token: ?[]const u8 = null;

/// Every control text the template spells; the file's other control
/// tokens share the shape `<|name|>`, which `markerAt` matches. Names and
/// call arguments carrying one are refused; the assistant's own past text
/// loses them.
const markers = [_][]const u8{ user, model, system, tool_role, text_part, thinking_part, call_part, xml_part, image_part, end, end_sampling, image_placeholder.? };
/// The deepest JSON nesting rendered or parsed: serialization recurses
/// once a level, so anything deeper is refused before it starts.
const max_depth = 64;

/// The template's effort table, as it prints each value.
pub fn effortLevel(effort: Effort) []const u8 {
    return switch (effort) {
        .off => "0",
        .low => "0.2",
        .medium => "0.7",
        .high => "0.9",
        .xhigh => "0.99",
    };
}

/// The bytes every rendering that starts with these system messages begins
/// with: the tool declarations, the system messages, and the effort line.
/// Pinned as a byte prefix of `render`.
pub fn prefix(alloc: std.mem.Allocator, messages: []const Message, tools: []const profiles.ToolDefinition, effort: Effort, limits: Limits) Error![]u8 {
    for (tools) |t| try checkName(t.name);
    var builder: Builder = .init(alloc, limits);
    errdefer builder.deinit();
    try renderHead(&builder, alloc, messages[0..profiles.leadingSystemCount(messages)], tools, effort);
    return builder.toOwnedSlice();
}

/// Returns an owned UTF-8 prompt; caller frees it with the supplied
/// allocator. Message strings are borrowed only for this call. Nothing is
/// trimmed: the template writes content and reasoning verbatim. A developer
/// message is a system message (the template knows no such role).
pub fn render(alloc: std.mem.Allocator, messages: []const Message, tools: []const profiles.ToolDefinition, effort: Effort, limits: Limits) Error![]u8 {
    try profiles.validate(alloc, messages, tools, limits);
    for (tools) |t| try checkName(t.name);
    const prefix_count = profiles.leadingSystemCount(messages);
    var builder: Builder = .init(alloc, limits);
    errdefer builder.deinit();
    try renderHead(&builder, alloc, messages[0..prefix_count], tools, effort);
    // The calls of the last assistant message: a result names its call's tool.
    var pending: []const profiles.ToolCall = &.{};
    for (messages[prefix_count..]) |message| switch (message.role) {
        .system, .developer => unreachable, // `validate`: only leading
        .user => {
            // One image message per image, before the text; the engine's
            // span overwrites the placeholders with projector rows.
            for (message.images) |image| {
                try builder.add(user ++ image_part);
                const count = @as(usize, image.width_tokens) * image.height_tokens;
                for (0..count) |_| try builder.add(image_placeholder.?);
                try builder.add(end);
            }
            if (message.content.len != 0 or message.images.len == 0) {
                try builder.add(user ++ text_part);
                try builder.addLiteral(message.content);
                try builder.add(end);
            }
        },
        .assistant => {
            // The model's own past text may spell a marker in ordinary
            // tokens; re-encoding it would make it a control token.
            const own_reasoning = try stripMarkers(alloc, message.reasoning_content);
            defer alloc.free(own_reasoning);
            const own_content = try stripMarkers(alloc, message.content);
            defer alloc.free(own_content);
            if (message.reasoning_content.len != 0) {
                try builder.add(model ++ thinking_part);
                try builder.add(own_reasoning);
                try builder.add(end);
            }
            if (message.content.len != 0 or (message.tool_calls.len == 0 and message.reasoning_content.len == 0)) {
                try builder.add(model ++ text_part);
                try builder.add(own_content);
                try builder.add(end);
            }
            for (message.tool_calls) |call| try renderCall(&builder, alloc, call);
            try builder.add(end_sampling);
            pending = message.tool_calls;
        },
        .tool => {
            // `validate` proved the id answers one of the pending calls.
            const name = for (pending) |call| {
                if (call.id == message.tool_call_id.?) break call.name;
            } else unreachable;
            try builder.add(tool_role);
            try builder.add(name);
            try builder.add(text_part);
            try builder.addLiteral(message.content);
            try builder.add(end);
        },
    };
    return builder.toOwnedSlice();
}

/// The tool declarations, the leading system messages, and the effort line,
/// which the template writes before the first other message.
fn renderHead(builder: *Builder, alloc: std.mem.Allocator, leading: []const Message, tools: []const profiles.ToolDefinition, effort: Effort) Error!void {
    if (tools.len != 0) {
        var out: std.Io.Writer.Allocating = .init(alloc);
        defer out.deinit();
        writeDeclarations(&out.writer, alloc, tools) catch |err| return if (err == error.UnsupportedContent) error.UnsupportedContent else jsonError(err);
        try builder.add(system ++ "tool_declare" ++ xml_part);
        try builder.addLiteral(out.written());
        try builder.add(end);
    }
    for (leading) |message| {
        try builder.add(system ++ text_part);
        try builder.addLiteral(message.content);
        try builder.add(end);
    }
    try builder.add(system ++ text_part ++ "Thinking effort level: ");
    try builder.add(effortLevel(effort));
    try builder.add(end);
}

/// `[{"description":…,"name":…,"parameters":…,"type":"function"},…]`, the
/// template's spec objects with every key sorted.
fn writeDeclarations(w: *std.Io.Writer, alloc: std.mem.Allocator, tools: []const profiles.ToolDefinition) !void {
    try w.writeByte('[');
    for (tools, 0..) |t, i| {
        if (i != 0) try w.writeByte(',');
        if (!try shallow(alloc, t.parameters)) return error.UnsupportedContent;
        const parsed = try std.json.parseFromSlice(std.json.Value, alloc, t.parameters, .{ .parse_numbers = false });
        defer parsed.deinit();
        try w.writeAll("{\"description\":");
        try std.json.Stringify.encodeJsonString(t.description, .{}, w);
        try w.writeAll(",\"name\":");
        try std.json.Stringify.encodeJsonString(t.name, .{}, w);
        try w.writeAll(",\"parameters\":");
        try writeJson(w, alloc, parsed.value, true);
        try w.writeAll(",\"type\":\"function\"}");
    }
    try w.writeByte(']');
}

/// One call message. The assistant's own bytes, so unframed: the name and
/// arguments may carry no marker.
fn renderCall(builder: *Builder, alloc: std.mem.Allocator, call: profiles.ToolCall) Error!void {
    try checkName(call.name);
    if (!try shallow(alloc, call.arguments)) return error.UnsupportedContent;
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    writeCall(&out.writer, alloc, call) catch |err| return jsonError(err);
    if (hasMarker(out.written())) return error.UnsupportedContent;
    try builder.add(model);
    try builder.add(call.name);
    try builder.add(call_part);
    try builder.add(out.written());
    try builder.add(end);
}

/// `{"name":NAME,"args":{…}}`: the arguments' keys sorted by code point,
/// each value compact in its own order, as the template's `dictsort` and
/// `tojson` write them.
fn writeCall(w: *std.Io.Writer, alloc: std.mem.Allocator, call: profiles.ToolCall) !void {
    // `validate` already proved these arguments are a JSON object.
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, call.arguments, .{ .parse_numbers = false });
    defer parsed.deinit();
    try w.writeAll("{\"name\":");
    try std.json.Stringify.encodeJsonString(call.name, .{}, w);
    try w.writeAll(",\"args\":{");
    const keys = try sortedKeys(alloc, parsed.value.object);
    defer alloc.free(keys);
    for (keys, 0..) |key, i| {
        if (i != 0) try w.writeByte(',');
        try std.json.Stringify.encodeJsonString(key, .{}, w);
        try w.writeByte(':');
        try writeJson(w, alloc, parsed.value.object.get(key).?, false);
    }
    try w.writeAll("}}");
}

/// A JSON value as `tojson` with compact separators writes it: keys sorted
/// at every level when `sorted`, else in their order; strings with Python's
/// escapes (non-ASCII literal); numbers as written.
fn writeJson(w: *std.Io.Writer, alloc: std.mem.Allocator, value: std.json.Value, sorted: bool) !void {
    switch (value) {
        .null => try w.writeAll("null"),
        .bool => |b| try w.writeAll(if (b) "true" else "false"),
        .integer => |n| try w.print("{d}", .{n}),
        .float => |f| try w.print("{d}", .{f}),
        .number_string => |text| try w.writeAll(text),
        .string => |text| try std.json.Stringify.encodeJsonString(text, .{}, w),
        .array => |array| {
            try w.writeByte('[');
            for (array.items, 0..) |item, i| {
                if (i != 0) try w.writeByte(',');
                try writeJson(w, alloc, item, sorted);
            }
            try w.writeByte(']');
        },
        .object => |object| {
            const keys = if (sorted) try sortedKeys(alloc, object) else try alloc.dupe([]const u8, object.keys());
            defer alloc.free(keys);
            try w.writeByte('{');
            for (keys, 0..) |key, i| {
                if (i != 0) try w.writeByte(',');
                try std.json.Stringify.encodeJsonString(key, .{}, w);
                try w.writeByte(':');
                try writeJson(w, alloc, object.get(key).?, sorted);
            }
            try w.writeByte('}');
        },
    }
}

/// Whether `text`'s JSON nests at most `max_depth` levels, read without
/// recursion; text that is not JSON passes, for the parser to refuse.
fn shallow(alloc: std.mem.Allocator, text: []const u8) std.mem.Allocator.Error!bool {
    var scanner = std.json.Scanner.initCompleteInput(alloc, text);
    defer scanner.deinit();
    var depth: usize = 0;
    while (true) {
        const token = scanner.next() catch |err| return if (err == error.OutOfMemory) error.OutOfMemory else true;
        switch (token) {
            .object_begin, .array_begin => {
                depth += 1;
                if (depth > max_depth) return false;
            },
            .object_end, .array_end => depth -|= 1,
            .end_of_document => return true,
            else => {},
        }
    }
}

/// The object's keys by code point (UTF-8 byte order is code point order).
/// Owned by `alloc`.
fn sortedKeys(alloc: std.mem.Allocator, object: std.json.ObjectMap) ![][]const u8 {
    const keys = try alloc.dupe([]const u8, object.keys());
    std.mem.sort([]const u8, keys, {}, struct {
        fn less(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.less);
    return keys;
}

/// `validate` already accepted this JSON, so a failure here is an
/// allocation failure (an allocating writer fails only that way).
fn jsonError(err: anyerror) Error {
    return switch (err) {
        error.OutOfMemory, error.WriteFailed => error.OutOfMemory,
        else => error.InvalidConversation,
    };
}

/// A tool name is spelled bare after a role token, so it may carry no
/// control text; nothing else in the template constrains it.
fn checkName(name: []const u8) Error!void {
    if (name.len == 0 or hasMarker(name)) return error.UnsupportedContent;
}

/// Parses one call body, the text after `<|content_invoke_tool_json|>`,
/// into an owned `ToolCall`, or null when it is not one JSON object holding
/// exactly a string `name` and an object `args`, nested at most
/// `max_depth` levels and spelling no control text (a call the history
/// could not render again). The arguments come back compact, in the
/// model's order.
pub fn parseTool(alloc: std.mem.Allocator, body: []const u8) std.mem.Allocator.Error!?profiles.ToolCall {
    const text = std.mem.trim(u8, body, " \t\r\n");
    if (!try shallow(alloc, text)) return null;
    const parsed = std.json.parseFromSlice(std.json.Value, alloc, text, .{ .parse_numbers = false }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    defer parsed.deinit();
    const object = switch (parsed.value) {
        .object => |o| o,
        else => return null,
    };
    if (object.count() != 2) return null;
    const name = switch (object.get("name") orelse return null) {
        .string => |s| s,
        else => return null,
    };
    checkName(name) catch return null;
    const args = object.get("args") orelse return null;
    if (args != .object) return null;
    const arguments = std.json.Stringify.valueAlloc(alloc, args, .{}) catch return error.OutOfMemory;
    errdefer alloc.free(arguments);
    if (hasMarker(arguments)) {
        alloc.free(arguments);
        return null;
    }
    const name_owned = try alloc.dupe(u8, name);
    return .{ .id = 0, .name = name_owned, .arguments = arguments };
}

/// The length of the control text `<|name|>` (name of `[a-z0-9_]`) at the
/// start of `text`, or 0.
fn markerAt(text: []const u8) usize {
    if (!std.mem.startsWith(u8, text, "<|")) return 0;
    var i: usize = 2;
    while (i < text.len and (std.ascii.isLower(text[i]) or std.ascii.isDigit(text[i]) or text[i] == '_')) i += 1;
    if (i == 2 or !std.mem.startsWith(u8, text[i..], "|>")) return 0;
    return i + 2;
}

fn hasMarker(text: []const u8) bool {
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, text, i, "<|")) |at| : (i = at + 1) {
        if (markerAt(text[at..]) != 0) return true;
    }
    return false;
}

/// `text` without any control marker; owned by `alloc`.
fn stripMarkers(alloc: std.mem.Allocator, text: []const u8) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    var i: usize = 0;
    while (i < text.len) {
        const skip = markerAt(text[i..]);
        if (skip != 0) {
            i += skip;
            continue;
        }
        try out.append(alloc, text[i]);
        i += 1;
    }
    return out.toOwnedSlice(alloc);
}

/// The file's `general.sampling.*` keys (temperature 1, top-k 0, top-p 1,
/// min-p 0), which the model's `generation_config.json` agrees with; one
/// setting for every effort.
pub fn samplingDefaults(effort: Effort) sampling.Options {
    _ = effort;
    return .{ .temperature = 1.0, .top_k = 0, .top_p = 1.0, .min_p = 0, .presence_penalty = 0, .repetition_penalty = 1 };
}

/// The mode's defaults with the caller's overrides applied per option.
pub fn samplingOptions(effort: Effort, overrides: SamplingOverrides) sampling.Options {
    return samplingDefaults(effort).override(overrides);
}

const Case = struct {
    name: []const u8,
    effort: Effort,
    messages: []const Message,
    tools: []const profiles.ToolDefinition = &.{},
    prompt: []const u8,
};
const Fixture = struct { template_sha256: []const u8, prompt_cases: []const Case };

fn expectFixture(json: []const u8, cases: usize) !void {
    const alloc = std.testing.allocator;
    const fixture = try std.json.parseFromSlice(Fixture, alloc, json, .{ .ignore_unknown_fields = true });
    defer fixture.deinit();
    try std.testing.expectEqualStrings(template_sha256, fixture.value.template_sha256);
    try std.testing.expectEqual(cases, fixture.value.prompt_cases.len);
    for (fixture.value.prompt_cases) |case| {
        errdefer std.debug.print("case {s}\n", .{case.name});
        const prompt = try render(alloc, case.messages, case.tools, case.effort, .{});
        defer alloc.free(prompt);
        try std.testing.expectEqualStrings(case.prompt, prompt);
    }
}

test "text prompts match the fixtures written from the template" {
    try expectFixture(@embedFile("fixtures/inkling2-text.json"), 9);
}

test "tool prompts match the fixtures written from the template" {
    try expectFixture(@embedFile("fixtures/inkling2-tools.json"), 5);
}

test "image prompts match the fixtures written from the template" {
    try expectFixture(@embedFile("fixtures/inkling2-media.json"), 2);
}

test "the effort line precedes the first other message and prints the template's table" {
    const alloc = std.testing.allocator;
    const want = [_][]const u8{ "0", "0.2", "0.7", "0.9", "0.99" };
    for (efforts, want) |effort, level| {
        const prompt = try render(alloc, &.{ .{ .role = .system, .content = "S" }, .{ .role = .developer, .content = "D" }, .{ .role = .user, .content = "U" } }, &.{}, effort, .{});
        defer alloc.free(prompt);
        const expected = try std.mem.concat(alloc, u8, &.{ system, text_part, "S", end, system, text_part, "D", end, system, text_part, "Thinking effort level: ", level, end, user, text_part, "U", end });
        defer alloc.free(expected);
        try std.testing.expectEqualStrings(expected, prompt);
    }
}

const read_tool: profiles.ToolDefinition = .{
    .name = "read_file",
    .description = "Read a file.",
    .parameters = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"offset\":{\"type\":\"integer\"}},\"required\":[\"path\"]}",
};

test "the prefix is a byte prefix of every rendering that starts with its system messages" {
    const alloc = std.testing.allocator;
    const sys: Message = .{ .role = .system, .content = "You are terse." };
    const dev: Message = .{ .role = .developer, .content = "Use Zig." };
    const first: Message = .{ .role = .user, .content = "hello" };
    for ([_]Effort{ .off, .high }) |effort| {
        for ([_][]const Message{ &.{ sys, dev, first }, &.{ sys, first }, &.{first} }) |messages| {
            for ([_][]const profiles.ToolDefinition{ &.{}, &.{read_tool} }) |tools| {
                const head = try prefix(alloc, messages[0 .. messages.len - 1], tools, effort, .{});
                defer alloc.free(head);
                const full = try render(alloc, messages, tools, effort, .{});
                defer alloc.free(full);
                try std.testing.expect(std.mem.startsWith(u8, full, head));
                try std.testing.expect(std.mem.startsWith(u8, full[head.len..], user ++ text_part));
            }
        }
    }
}

test "declarations sort every key; a call sorts its arguments' keys only" {
    const alloc = std.testing.allocator;
    const tools = [_]profiles.ToolDefinition{.{ .name = "f", .description = "", .parameters = "{\"z\":1,\"a\":{\"y\":\"é\\n\",\"b\":[{\"d\":0,\"c\":1.50}]}}" }};
    const call: profiles.ToolCall = .{ .id = 1, .name = "f", .arguments = "{\"b\":{\"z\":1,\"a\":2},\"B\":\"\\u0001/\",\"a\":[3, 4]}" };
    const prompt = try render(alloc, &.{
        .{ .role = .user, .content = "go" },
        .{ .role = .assistant, .content = "", .tool_calls = &.{call} },
        .{ .role = .tool, .content = "ok", .tool_call_id = 1 },
    }, &tools, .off, .{});
    defer alloc.free(prompt);
    try std.testing.expect(std.mem.startsWith(u8, prompt, system ++ "tool_declare" ++ xml_part ++
        "[{\"description\":\"\",\"name\":\"f\",\"parameters\":{\"a\":{\"b\":[{\"c\":1.50,\"d\":0}],\"y\":\"é\\n\"},\"z\":1},\"type\":\"function\"}]" ++ end));
    try std.testing.expect(std.mem.indexOf(u8, prompt, model ++ "f" ++ call_part ++
        "{\"name\":\"f\",\"args\":{\"B\":\"\\u0001/\",\"a\":[3,4],\"b\":{\"z\":1,\"a\":2}}}" ++ end ++ end_sampling ++ tool_role ++ "f" ++ text_part ++ "ok" ++ end) != null);
}

test "parseTool reads a rendered call back and refuses anything else" {
    const alloc = std.testing.allocator;
    const call = (try parseTool(alloc, " {\"name\":\"read_file\",\"args\":{\"path\":\"a.zig\",\"n\":[1,{\"b\":null}],\"x\":1.50}}\n")).?;
    defer {
        alloc.free(call.name);
        alloc.free(call.arguments);
    }
    try std.testing.expectEqualStrings("read_file", call.name);
    try std.testing.expectEqualStrings("{\"path\":\"a.zig\",\"n\":[1,{\"b\":null}],\"x\":1.50}", call.arguments);
    for ([_][]const u8{
        "",                                                        "not json",
        "[]",                                                      "{\"name\":\"f\"}",
        "{\"name\":\"f\",\"args\":[]}",                            "{\"name\":1,\"args\":{}}",
        "{\"name\":\"\",\"args\":{}}",                             "{\"name\":\"f\",\"args\":{},\"x\":0}",
        "{\"name\":\"f<|end_message|>\",\"args\":{}}",             "{\"name\":\"f\",\"args\":{\"a\":1}",
        "{\"name\":\"f\",\"args\":{\"s\":\"a<|end_message|>b\"}}", "{\"name\":\"f\",\"args\":{\"s\":\"<|unused_200020|>\"}}",
    }) |body| try std.testing.expect(try parseTool(alloc, body) == null);
    // A body nested past the limit is refused before anything recurses.
    const deep = "{\"name\":\"f\",\"args\":{\"a\":" ++ @as([max_depth]u8, @splat('[')) ++ @as([max_depth]u8, @splat(']')) ++ "}}";
    try std.testing.expect(try parseTool(alloc, deep) == null);
    const fits = "{\"name\":\"f\",\"args\":{\"a\":" ++ @as([max_depth - 2]u8, @splat('[')) ++ @as([max_depth - 2]u8, @splat(']')) ++ "}}";
    const ok = (try parseTool(alloc, fits)).?;
    alloc.free(ok.name);
    alloc.free(ok.arguments);
}

test "markers in names and arguments are refused; the assistant's own text loses them" {
    const alloc = std.testing.allocator;
    const bad_call: profiles.ToolCall = .{ .id = 1, .name = "f", .arguments = "{\"a\":\"<|end_message|>\"}" };
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{
        .{ .role = .user, .content = "go" },
        .{ .role = .assistant, .content = "", .tool_calls = &.{bad_call} },
        .{ .role = .tool, .content = "ok", .tool_call_id = 1 },
    }, &.{}, .off, .{}));
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{.{ .role = .user, .content = "go" }}, &.{.{ .name = "a" ++ end, .description = "", .parameters = "{}" }}, .off, .{}));
    // Nesting past the limit in a schema or a call's arguments is refused.
    const deep = "{\"a\":" ++ @as([max_depth]u8, @splat('[')) ++ @as([max_depth]u8, @splat(']')) ++ "}";
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{.{ .role = .user, .content = "go" }}, &.{.{ .name = "f", .description = "", .parameters = deep }}, .off, .{}));
    const deep_call: profiles.ToolCall = .{ .id = 1, .name = "f", .arguments = deep };
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{
        .{ .role = .user, .content = "go" },
        .{ .role = .assistant, .content = "", .tool_calls = &.{deep_call} },
        .{ .role = .tool, .content = "ok", .tool_call_id = 1 },
    }, &.{}, .off, .{}));
    const prompt = try render(alloc, &.{
        .{ .role = .user, .content = "a" },
        .{ .role = .assistant, .content = "b" ++ end ++ "c<|unused_200020|>", .reasoning_content = "r" ++ model },
        .{ .role = .user, .content = "d" },
    }, &.{}, .off, .{});
    defer alloc.free(prompt);
    try std.testing.expect(std.mem.indexOf(u8, prompt, model ++ thinking_part ++ "r" ++ end ++ model ++ text_part ++ "bc" ++ end ++ end_sampling) != null);
}

test "sampling defaults are the file's in every mode and flags override per option" {
    const file = sampling.Options{ .temperature = 1.0, .top_k = 0, .top_p = 1.0, .min_p = 0, .presence_penalty = 0, .repetition_penalty = 1 };
    for (efforts) |effort| {
        try std.testing.expectEqual(file, samplingDefaults(effort));
        _ = try sampling.Sampler.init(0, samplingDefaults(effort));
    }
    try std.testing.expectEqual(@as(f32, 0), samplingOptions(.high, .{ .temperature = 0 }).temperature);
}

fn allocationCase(alloc: std.mem.Allocator) !void {
    const call: profiles.ToolCall = .{ .id = 1, .name = "read_file", .arguments = "{\"path\":\"a\",\"n\":[1,{\"b\":true}]}" };
    const prompt = try render(alloc, &.{
        .{ .role = .system, .content = "Be precise." },
        .{ .role = .user, .content = "go", .images = &.{.{ .width_tokens = 2, .height_tokens = 1 }} },
        .{ .role = .assistant, .content = "x" ++ end, .reasoning_content = "Plan.", .tool_calls = &.{call} },
        .{ .role = .tool, .content = "body", .tool_call_id = 1 },
    }, &.{read_tool}, .medium, .{});
    defer alloc.free(prompt);
    const head = try prefix(alloc, &.{.{ .role = .system, .content = "Be precise." }}, &.{read_tool}, .xhigh, .{});
    defer alloc.free(head);
    if (try parseTool(alloc, "{\"name\":\"f\",\"args\":{\"a\":[1,{\"c\":null}]}}")) |parsed| {
        alloc.free(parsed.name);
        alloc.free(parsed.arguments);
    } else return error.UnexpectedNull;
}

test "prompt rendering and call parsing clean up every allocation failure" {
    try alloc_check.checkAll(std.testing.allocator, allocationCase, .{});
}

const Sink = struct {
    alloc: std.mem.Allocator,
    thinking: std.Io.Writer.Allocating,
    answer: std.Io.Writer.Allocating,
    calls: std.ArrayList(profiles.ToolCall) = .empty,
    cut: ?usize = null,

    fn init(alloc: std.mem.Allocator) Sink {
        return .{ .alloc = alloc, .thinking = .init(alloc), .answer = .init(alloc) };
    }
    fn deinit(self: *Sink) void {
        for (self.calls.items) |c| {
            self.alloc.free(c.name);
            self.alloc.free(c.arguments);
        }
        self.calls.deinit(self.alloc);
        self.thinking.deinit();
        self.answer.deinit();
    }
    pub fn send(self: *Sink, e: @import("../events.zig").Event) !void {
        switch (e) {
            .thinking => |t| try self.thinking.writer.writeAll(t),
            .answer => |t| try self.answer.writer.writeAll(t),
            .tool_call => |c| try self.calls.append(self.alloc, .{ .id = 0, .name = try self.alloc.dupe(u8, c.name), .arguments = try self.alloc.dupe(u8, c.arguments) }),
            .tool_progress => {},
            .tool_cut => |c| self.cut = c.bytes,
            .stop => {},
        }
    }
};

/// A vocabulary of the bytes and the template's control tokens, ids from 256
/// in `markers` order, so `Profile.decoder` resolves them as on the file.
fn testVocabulary(alloc: std.mem.Allocator) !@import("../tokenizer/vocabulary.zig").Vocabulary {
    const vocabulary = @import("../tokenizer/vocabulary.zig");
    var arena: std.heap.ArenaAllocator = .init(alloc);
    errdefer arena.deinit();
    const a = arena.allocator();
    var tokens: std.ArrayList(vocabulary.Token) = .empty;
    var ids: std.StringHashMapUnmanaged(u32) = .empty;
    for (0..256) |byte| {
        const text = try std.fmt.allocPrint(a, "<0x{X:0>2}>", .{byte});
        try ids.put(a, text, @intCast(tokens.items.len));
        try tokens.append(a, .{ .text = text, .kind = .byte });
    }
    for (markers) |text| {
        try ids.put(a, text, @intCast(tokens.items.len));
        try tokens.append(a, .{ .text = text, .kind = .control });
    }
    return .{ .storage = arena, .tokens = tokens.items, .pre = "inkling2", .bos = null, .eos = null, .padding = null, .token_ids = ids, .merge_ranks = .empty };
}

fn markerId(text: []const u8) u32 {
    for (markers, 0..) |m, i| if (std.mem.eql(u8, m, text)) return @intCast(256 + i);
    unreachable;
}

test "the decoder routes each segment by its content type at every effort" {
    const alloc = std.testing.allocator;
    var vocab = try testVocabulary(alloc);
    defer vocab.deinit();
    const outcome: @import("../engine.zig").Outcome = .{ .stop = .eos, .timing = .{ .prompt_tokens = 1, .generated_tokens = 1 } };
    // The stream a greedy reference produced at effort 0.2, and the same at
    // effort off, which the model card says does not stop reasoning.
    for (efforts) |effort| {
        var sink = Sink.init(alloc);
        defer sink.deinit();
        var d = try profiles.Profile.inkling2.decoder(alloc, &vocab, effort, &.{markerId(end)});
        defer d.deinit();
        try d.feed(markerId(model), model, &sink);
        try d.feed(markerId(thinking_part), thinking_part, &sink);
        try d.feed(1, "The capital", &sink);
        try d.feed(1, " is Paris.", &sink);
        // A budget cut closes the reasoning with `<|end_message|>`.
        try std.testing.expectEqual(@as(?u32, markerId(end)), d.cutToken(10, 1));
        try d.feed(markerId(end), end, &sink);
        // Between segments nothing is cut: a header is being read.
        try std.testing.expectEqual(@as(?u32, null), d.cutToken(10, 1));
        try d.feed(markerId(model), model, &sink);
        try d.feed(markerId(text_part), text_part, &sink);
        try d.feed(1, "Par", &sink);
        try d.feed(1, "is", &sink);
        try d.feed(markerId(end), end, &sink);
        try d.end(outcome, &sink);
        try std.testing.expectEqualStrings("The capital is Paris.", sink.thinking.written());
        try std.testing.expectEqualStrings("Paris", sink.answer.written());
    }
}

test "the decoder delivers a call named in its header and releases a malformed one as text" {
    const alloc = std.testing.allocator;
    var vocab = try testVocabulary(alloc);
    defer vocab.deinit();
    const outcome: @import("../engine.zig").Outcome = .{ .stop = .eos, .timing = .{ .prompt_tokens = 1, .generated_tokens = 1 } };
    var sink = Sink.init(alloc);
    defer sink.deinit();
    var d = try profiles.Profile.inkling2.decoder(alloc, &vocab, .high, &.{});
    defer d.deinit();
    try d.feed(markerId(model), model, &sink);
    try d.feed(markerId(thinking_part), thinking_part, &sink);
    try d.feed(1, "Read it.", &sink);
    try std.testing.expectEqual(@as(?u32, markerId(end)), d.cutToken(10, 1));
    try d.feed(markerId(end), end, &sink);
    try d.feed(markerId(model), model, &sink);
    try d.feed(1, "read_", &sink);
    try d.feed(1, "file", &sink);
    try d.feed(markerId(call_part), call_part, &sink);
    for ("{\"name\":\"read_file\",\"args\":{\"path\":\"a.zig\"}}") |byte| try d.feed(1, &.{byte}, &sink);
    try d.feed(markerId(end), end, &sink);
    try d.feed(markerId(model), model, &sink);
    try d.feed(markerId(call_part), call_part, &sink);
    try d.feed(1, "{\"name\":\"bash\",\"args\":{\"command\":\"ls\"}}", &sink);
    // The stop token ends the turn with a second call still unclosed.
    try d.end(outcome, &sink);
    try std.testing.expectEqualStrings("Read it.", sink.thinking.written());
    try std.testing.expectEqualStrings("", sink.answer.written());
    try std.testing.expectEqual(@as(usize, 2), sink.calls.items.len);
    try std.testing.expectEqualStrings("read_file", sink.calls.items[0].name);
    try std.testing.expectEqualStrings("{\"path\":\"a.zig\"}", sink.calls.items[0].arguments);
    try std.testing.expectEqualStrings("bash", sink.calls.items[1].name);

    var bad = Sink.init(alloc);
    defer bad.deinit();
    var e = try profiles.Profile.inkling2.decoder(alloc, &vocab, .off, &.{});
    defer e.deinit();
    try e.feed(markerId(model), model, &bad);
    try e.feed(markerId(call_part), call_part, &bad);
    try e.feed(1, "not a call", &bad);
    try e.feed(markerId(end), end, &bad);
    try e.end(outcome, &bad);
    try std.testing.expectEqual(@as(usize, 0), bad.calls.items.len);
    try std.testing.expectEqualStrings("not a call", bad.answer.written());
}
