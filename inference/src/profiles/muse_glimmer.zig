//! Bounded prompt profile for the pinned Muse Glimmer 30B artifact,
//! implemented from the chat template embedded in the GGUF and verified
//! against prompts the pinned llama.cpp server rendered from it. The template
//! text is not in the tree; `fixtures/muse_glimmer-text.json` is its evidence.
//! The behavior it encodes — `<|start|>ROLE<|message|>…<|eot|>` turns, one
//! system turn per leading system message (or a synthesized one), the
//! reasoning-strength line, reasoning kept as its own `to=self` message —
//! is specified with fixture-backed clauses in docs/reference/prompt-profile.md.
//!
//! Two deviations, both decided: the synthesized system turn omits the
//! template's `Current date:` line (the profile takes no clock; the tests
//! remove the captured line), and `Effort.off` renders as `low` (the model
//! always opens its reasoning message).
//!
//! Tool calling is the template's ATEM protocol (`fixtures/muse_glimmer-tools.json`,
//! docs/reference/tool-calling.md): declarations as JSON lines in the
//! system turn, each call its own `assistant to=NAME` message holding an
//! `<atem:function_calls>` block, each result its own `tool NAME` turn.
//! The stream decoder's channel grammar hands a `to=NAME` body to `parseTool`.
//! It owns no tokenizer or model equations.
const std = @import("std");
const sampling = @import("../sampling/root.zig");

/// Exact GGUF template this profile implements (`tokenizer.chat_template`,
/// 7,167 bytes, no trailing newline); `profiles.forDocument` selects the
/// profile by this digest, never by architecture.
pub const template_sha256 = "114f55ebdc1804c1af371197b9fdf2d6bb925966c9dfe46b73782a71bc07965e";
/// Other revisions proven to render every fixture identically; none so far
/// (the Hub repository's `chat_template.jinja` differs and is unchecked).
pub const template_aliases = [_][]const u8{};

const profiles = @import("root.zig");
pub const Role = profiles.Role;
pub const Effort = profiles.Effort;
pub const Message = profiles.Message;
pub const Limits = profiles.Limits;
pub const Error = profiles.Error;
pub const SamplingOverrides = profiles.SamplingOverrides;

/// The tokens that end a generation: `<|eot|>` (200008, the file's EOT)
/// and `<|end_of_text|>` (200001, its EOS). `<|eom|>` ends a message, not
/// the turn.
pub const stop_tokens = [_][]const u8{ "<|eot|>", "<|end_of_text|>" };

/// How the model delimits its reasoning: a message addressed to itself.
/// The generation prompt ends at `<|start|>assistant`, so the first header
/// (` to=self`) arrives as ordinary text and the decoder routes on it.
/// The image span placeholder, inside `<|image_start|>` … `<|image_end|>`.
pub const image_placeholder: ?[]const u8 = "<|patch|>";
pub const reasoning: profiles.Reasoning = .{ .open = "<|start|>assistant to=self<|message|>", .close = "<|eom|>" };
pub const stream_markers: profiles.StreamMarkers = .{
    .open = "<|start|>",
    .close = "<|eom|>",
    .tool_parse = parseTool,
    .channel = .{ .start = "<|start|>", .message = "<|message|>", .eom = "<|eom|>" },
};

/// The template's opening text: the encoder never adds BOS, so the profile
/// writes it (the reference server's `/apply-template` strips it).
const bos = "<|begin_of_text|>";
/// The system turn the template synthesizes when no system message exists,
/// without its date line.
const default_system = "You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.";
/// Every control marker the template spells, and the ATEM markup the
/// reference parses with regular expressions. Content, results, names, and
/// arguments carrying one are rejected rather than rendered, so structure
/// cannot be smuggled past the profile; the assistant's own past text loses
/// them instead.
const markers = [_][]const u8{
    "<|start|>",             "<|message|>",            "<|eom|>",         "<|eot|>",        "<|end_of_text|>", bos,
    "<atem:function_calls>", "</atem:function_calls>", "<atem:invoke",    "</atem:invoke>", "<atem:parameter", "</atem:parameter>",
    "<tool_output",          "</tool_output>",         "<|image_start|>", "<|image_end|>",  "<|patch|>",
};
/// The template's tool instructions, verbatim, around the declarations.
const tools_intro =
    "In this environment you have access to a set of tools you can use to answer the user's question.\n\n" ++
    "You can invoke a function by writing a \"<atem:function_calls>\" block like the following:\n" ++
    "<atem:function_calls>\n<atem:invoke name=\"$FUNCTION_NAME\">\n<atem:parameter name=\"$PARAMETER_NAME\">$PARAMETER_VALUE</atem:parameter>\n...\n</atem:invoke>\n</atem:function_calls>\n\n" ++
    "String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.\n" ++
    "Here are the functions available in JSONSchema format:\n" ++
    "// Tool metadata\n";
const tools_example =
    "\n\nHere's an example of how to call a function in the tool set:\n" ++
    "(If the tool namespace is not specified, invoke the function directly as `example_function_name` rather than `example_tool_name.example_function_name`)\n\n" ++
    "to=example_tool_name.example_function_name\n\n" ++
    "<atem:function_calls>\n<atem:invoke name=\"example_tool_name.example_function_name\">\n" ++
    "<atem:parameter name=\"example_parameter_1\">value_1</atem:parameter>\n" ++
    "<atem:parameter name=\"example_parameter_2\">This is the value for the second parameter\nthat can span\n\"multiple\" lines\n</atem:parameter>\n" ++
    "</atem:invoke>\n</atem:function_calls>";

/// The bytes every rendering that starts with these system messages begins
/// with: `<|begin_of_text|>` and the system turns (the synthesized one when
/// there are none). Pinned as a byte prefix of `render`.
pub fn prefix(alloc: std.mem.Allocator, messages: []const Message, tools: []const profiles.ToolDefinition, effort: Effort, limits: Limits) Error![]u8 {
    for (tools) |tool| try checkName(tool.name);
    var builder: Builder = .{ .alloc = alloc, .limit = limits.output_bytes };
    errdefer builder.bytes.deinit(alloc);
    try renderSystem(&builder, alloc, messages[0..profiles.leadingSystemCount(messages)], tools, effort);
    return builder.bytes.toOwnedSlice(alloc);
}

/// Returns an owned UTF-8 prompt; caller frees it with the supplied allocator.
/// All message strings are borrowed only for this call. `profiles.validate`
/// enforces the shared conversation rules. Nothing is trimmed: the template
/// writes content and reasoning verbatim, and the model's own reasoning
/// bytes end where `<|eom|>` begins.
pub fn render(alloc: std.mem.Allocator, messages: []const Message, tools: []const profiles.ToolDefinition, effort: Effort, limits: Limits) Error![]u8 {
    try profiles.validate(alloc, messages, tools, limits);
    for (tools) |tool| try checkName(tool.name);
    const prefix_count = profiles.leadingSystemCount(messages);
    var builder: Builder = .{ .alloc = alloc, .limit = limits.output_bytes };
    errdefer builder.bytes.deinit(alloc);
    try renderSystem(&builder, alloc, messages[0..prefix_count], tools, effort);
    // The calls of the last assistant message: a result names its call's tool.
    var pending: []const profiles.ToolCall = &.{};
    for (messages[prefix_count..]) |message| switch (message.role) {
        .system, .developer => unreachable, // `validate`: only leading
        .user => {
            if (hasMarker(message.content)) return error.UnsupportedContent;
            try builder.add("<|start|>user<|message|>");
            // The reference's markers around `count` placeholders the
            // engine's spans overwrite with projector rows, one run per
            // image, before the text and without newlines.
            for (message.images) |image| {
                try builder.add("<|image_start|>");
                var n: usize = 0;
                const count = @as(usize, image.width_tokens) * image.height_tokens;
                while (n < count) : (n += 1) try builder.add(image_placeholder.?);
                try builder.add("<|image_end|>");
            }
            try builder.add(message.content);
            try builder.add("<|eot|>");
        },
        .assistant => {
            // The model's own past text may carry marker text (a header
            // released as text, a truncated message): re-encoding it would
            // turn text into control tokens, so the markers are removed.
            const own_reasoning = try stripMarkers(alloc, message.reasoning_content);
            defer alloc.free(own_reasoning);
            const own_content = try stripMarkers(alloc, message.content);
            defer alloc.free(own_content);
            if (own_reasoning.len != 0) {
                try builder.add(reasoning.open);
                try builder.add(own_reasoning);
                try builder.add(reasoning.close);
            }
            if (message.tool_calls.len != 0) {
                // A call is its own message; the content is not rendered.
                // `validate` proved the results follow, so the last call
                // ends the turn.
                for (message.tool_calls, 0..) |call, i| {
                    try renderCall(&builder, alloc, call);
                    try builder.add(if (i + 1 == message.tool_calls.len) "<|eot|>" else "<|eom|>");
                }
                pending = message.tool_calls;
                continue;
            }
            try builder.add("<|start|>assistant to=user<|message|>");
            try builder.add(own_content);
            try builder.add("<|eot|>");
        },
        .tool => {
            if (hasMarker(message.content)) return error.UnsupportedContent;
            // `validate` proved the id answers one of the pending calls.
            const name = for (pending) |call| {
                if (call.id == message.tool_call_id.?) break call.name;
            } else unreachable;
            try builder.add("<|start|>tool ");
            try builder.add(name);
            try builder.add("<|message|><tool_output name=\"");
            try builder.add(name);
            try builder.add("\">\n");
            try builder.add(message.content);
            try builder.add("\n</tool_output><|eot|>");
        },
    };
    try builder.add("<|start|>assistant");
    return builder.bytes.toOwnedSlice(alloc);
}

/// `<|begin_of_text|>` and one system turn per leading system or developer
/// message (the reference renders both roles as `system`), each closed by
/// the strength and recipients lines; the synthesized turn when there are
/// none. Content is verbatim: the template does not trim.
fn renderSystem(builder: *Builder, alloc: std.mem.Allocator, leading: []const Message, tools: []const profiles.ToolDefinition, effort: Effort) Error!void {
    try builder.add(bos);
    if (leading.len == 0) {
        try builder.add("<|start|>system<|message|>");
        try builder.add(default_system);
        try renderSystemMeta(builder, alloc, tools, effort);
        return;
    }
    for (leading) |message| {
        if (hasMarker(message.content)) return error.UnsupportedContent;
        try builder.add("<|start|>system<|message|>");
        try builder.add(message.content);
        try renderSystemMeta(builder, alloc, tools, effort);
    }
}

/// The lines the template appends to every system turn: the strength, the
/// tool declarations when there are any, and the recipients (`self`, one
/// `NS.*` per tool namespace in first-seen order, `user`).
fn renderSystemMeta(builder: *Builder, alloc: std.mem.Allocator, tools: []const profiles.ToolDefinition, effort: Effort) Error!void {
    try builder.add("\n\nReasoning strength: ");
    try builder.add(strengthName(effort));
    try builder.add(".");
    if (tools.len != 0) {
        try builder.add("\n\n");
        try renderDeclarations(builder, alloc, tools);
    }
    try builder.add("\n\n# Valid recipients: \"self\"");
    for (tools, 0..) |tool, i| {
        const ns = namespaceOf(tool.name);
        if (!firstNamespace(tools[0..i], ns)) continue;
        try builder.add(", \"");
        try builder.add(ns);
        try builder.add(".*\"");
    }
    try builder.add(", \"user\".<|eot|>");
}

/// A tool's namespace: its name before the first `.`, or the whole name.
fn namespaceOf(name: []const u8) []const u8 {
    return name[0 .. std.mem.indexOfScalar(u8, name, '.') orelse name.len];
}

/// Whether no earlier tool shares the namespace.
fn firstNamespace(earlier: []const profiles.ToolDefinition, ns: []const u8) bool {
    for (earlier) |tool| if (std.mem.eql(u8, namespaceOf(tool.name), ns)) return false;
    return true;
}

/// The template's `render_tool_defs`: the instructions, one metadata line
/// per namespace (empty description), one schema line per tool in the
/// reference's `tojson` style, and the fixed example.
fn renderDeclarations(builder: *Builder, alloc: std.mem.Allocator, tools: []const profiles.ToolDefinition) Error!void {
    try builder.add(tools_intro);
    for (tools, 0..) |tool, i| {
        const ns = namespaceOf(tool.name);
        if (!firstNamespace(tools[0..i], ns)) continue;
        try builder.add("{\"name\": ");
        try addJsonString(builder, alloc, ns);
        try builder.add(", \"description\": \"\"}\n");
    }
    try builder.add("// Function schemas");
    for (tools) |tool| {
        try builder.add("\n{\"name\": ");
        try addJsonString(builder, alloc, tool.name);
        try builder.add(", \"description\": ");
        try addJsonString(builder, alloc, tool.description);
        try builder.add(", \"parameters\": ");
        // `validate` already proved this is a JSON object.
        const parsed = std.json.parseFromSlice(std.json.Value, alloc, tool.parameters, .{ .parse_numbers = false }) catch |err| return jsonError(err);
        defer parsed.deinit();
        try addJson(builder, alloc, parsed.value);
        try builder.add("}");
    }
    try builder.add(tools_example);
}

/// One call as its own message: `<|start|>assistant to=NAME<|message|>`
/// and the ATEM block, scalars as is (strings verbatim, `true`/`false`/
/// `null`, numbers as written), lists and objects as JSON.
fn renderCall(builder: *Builder, alloc: std.mem.Allocator, call: profiles.ToolCall) Error!void {
    try checkName(call.name);
    try builder.add("<|start|>assistant to=");
    try builder.add(call.name);
    try builder.add("<|message|><atem:function_calls>\n<atem:invoke name=\"");
    try builder.add(call.name);
    try builder.add("\">\n");
    // `validate` already proved these arguments are a JSON object.
    const parsed = std.json.parseFromSlice(std.json.Value, alloc, call.arguments, .{ .parse_numbers = false }) catch |err| return jsonError(err);
    defer parsed.deinit();
    var it = parsed.value.object.iterator();
    while (it.next()) |entry| {
        try checkName(entry.key_ptr.*);
        try builder.add("<atem:parameter name=\"");
        try builder.add(entry.key_ptr.*);
        try builder.add("\">");
        switch (entry.value_ptr.*) {
            .string => |text| {
                if (hasMarker(text)) return error.UnsupportedContent;
                try builder.add(text);
            },
            .array, .object => try addJson(builder, alloc, entry.value_ptr.*),
            else => try addJson(builder, alloc, entry.value_ptr.*),
        }
        try builder.add("</atem:parameter>\n");
    }
    try builder.add("</atem:invoke>\n</atem:function_calls>");
}

/// `validate` already accepted this JSON, so a parse failure here is an
/// allocation failure and must not be misreported.
fn jsonError(err: anyerror) Error {
    return if (err == error.OutOfMemory) error.OutOfMemory else error.InvalidConversation;
}

/// A JSON string in the reference's `tojson` style (quotes, backslashes,
/// and control characters escaped; everything else, non-ASCII included,
/// literal).
fn addJsonString(builder: *Builder, alloc: std.mem.Allocator, text: []const u8) Error!void {
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    std.json.Stringify.encodeJsonString(text, .{}, &out.writer) catch return error.OutOfMemory;
    try builder.add(out.written());
}

/// A JSON value in the reference's `tojson` style: `", "` between items,
/// `": "` after keys, keys in insertion order, numbers as written.
fn addJson(builder: *Builder, alloc: std.mem.Allocator, value: std.json.Value) Error!void {
    switch (value) {
        .null => try builder.add("null"),
        .bool => |b| try builder.add(if (b) "true" else "false"),
        .integer => |n| {
            var buf: [24]u8 = undefined;
            try builder.add(std.fmt.bufPrint(&buf, "{d}", .{n}) catch unreachable);
        },
        .float => |f| {
            var buf: [64]u8 = undefined;
            try builder.add(std.fmt.bufPrint(&buf, "{d}", .{f}) catch return error.UnsupportedContent);
        },
        .number_string => |text| try builder.add(text),
        .string => |text| try addJsonString(builder, alloc, text),
        .array => |array| {
            try builder.add("[");
            for (array.items, 0..) |item, i| {
                if (i != 0) try builder.add(", ");
                try addJson(builder, alloc, item);
            }
            try builder.add("]");
        },
        .object => |object| {
            try builder.add("{");
            var it = object.iterator();
            var i: usize = 0;
            while (it.next()) |entry| : (i += 1) {
                if (i != 0) try builder.add(", ");
                try addJsonString(builder, alloc, entry.key_ptr.*);
                try builder.add(": ");
                try addJson(builder, alloc, entry.value_ptr.*);
            }
            try builder.add("}");
        },
    }
}

/// A tool or parameter name is spelled bare in `to=NAME` and inside
/// `name="…"`, so it can carry neither a quote, an angle bracket,
/// whitespace, nor a marker.
fn checkName(name: []const u8) Error!void {
    if (name.len == 0 or hasMarker(name)) return error.UnsupportedContent;
    for (name) |byte| switch (byte) {
        '"', '<', '>', ' ', '\t', '\r', '\n' => return error.UnsupportedContent,
        else => {},
    };
}

/// Parses one native call body — the text after `<|message|>` of an
/// `assistant to=NAME` message — into an owned `ToolCall`, or null when it
/// is not one complete, well-formed `<atem:function_calls>` block with a
/// single invoke. The inverse of `renderCall`: a parameter value that
/// parses as JSON keeps its type, anything else is a literal string (the
/// template's own rule for scalars), verbatim, spaces included.
pub fn parseTool(alloc: std.mem.Allocator, body: []const u8) std.mem.Allocator.Error!?profiles.ToolCall {
    var text = std.mem.trim(u8, body, " \t\r\n");
    text = take(text, "<atem:function_calls>") orelse return null;
    text = std.mem.trimStart(u8, text, " \t\r\n");
    text = take(text, "<atem:invoke name=\"") orelse return null;
    const name_end = std.mem.indexOfScalar(u8, text, '"') orelse return null;
    const name = text[0..name_end];
    checkName(name) catch return null;
    text = take(text[name_end..], "\">") orelse return null;

    var arguments: std.Io.Writer.Allocating = .init(alloc);
    // Also frees the partial JSON on every null return; the success path
    // takes ownership of the bytes first.
    defer arguments.deinit();
    var json: std.json.Stringify = .{ .writer = &arguments.writer };
    json.beginObject() catch return error.OutOfMemory;
    while (true) {
        text = std.mem.trimStart(u8, text, " \t\r\n");
        const param = take(text, "<atem:parameter name=\"") orelse break;
        const key_end = std.mem.indexOfScalar(u8, param, '"') orelse return null;
        const key = param[0..key_end];
        checkName(key) catch return null;
        const value_start = take(param[key_end..], "\">") orelse return null;
        const value_end = std.mem.indexOf(u8, value_start, "</atem:parameter>") orelse return null;
        json.objectField(key) catch return error.OutOfMemory;
        try writeArgumentValue(&json, alloc, value_start[0..value_end]);
        text = value_start[value_end + "</atem:parameter>".len ..];
    }
    text = take(text, "</atem:invoke>") orelse return null;
    text = std.mem.trimStart(u8, text, " \t\r\n");
    text = take(text, "</atem:function_calls>") orelse return null;
    if (std.mem.trim(u8, text, " \t\r\n").len != 0) return null;
    json.endObject() catch return error.OutOfMemory;
    const name_owned = try alloc.dupe(u8, name);
    errdefer alloc.free(name_owned);
    const arguments_owned = try arguments.toOwnedSlice();
    errdefer alloc.free(arguments_owned);
    return .{ .id = 0, .name = name_owned, .arguments = arguments_owned };
}

/// `text` after `literal`, or null when it does not start with it.
fn take(text: []const u8, literal: []const u8) ?[]const u8 {
    return if (std.mem.startsWith(u8, text, literal)) text[literal.len..] else null;
}

/// A parameter value on the wire: JSON when it parses as JSON, a literal
/// string otherwise.
fn writeArgumentValue(json: *std.json.Stringify, alloc: std.mem.Allocator, raw: []const u8) std.mem.Allocator.Error!void {
    if (raw.len > 0) {
        if (std.json.parseFromSlice(std.json.Value, alloc, raw, .{})) |parsed| {
            defer parsed.deinit();
            json.write(parsed.value) catch return error.OutOfMemory;
            return;
        } else |_| {}
    }
    json.write(raw) catch return error.OutOfMemory;
}

/// The template's `reasoning_strength`: the shared levels, with `off`
/// rendered as `low` since the model always reasons.
pub fn strengthName(effort: Effort) []const u8 {
    return switch (effort) {
        .off, .low => "low",
        .medium => "medium",
        .high => "high",
        .xhigh => "xhigh",
    };
}

fn hasMarker(text: []const u8) bool {
    for (markers) |marker| {
        if (std.mem.indexOf(u8, text, marker) != null) return true;
    }
    return false;
}

/// `text` without any control marker; owned by `alloc`.
fn stripMarkers(alloc: std.mem.Allocator, text: []const u8) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    var i: usize = 0;
    scan: while (i < text.len) {
        for (markers) |marker| if (std.mem.startsWith(u8, text[i..], marker)) {
            i += marker.len;
            continue :scan;
        };
        try out.append(alloc, text[i]);
        i += 1;
    }
    return out.toOwnedSlice(alloc);
}

/// The model card's "Best Practices" sampling parameters (read 2026-09-19
/// from `unsloth/Muse-Glimmer-30B-GGUF`: temperature 1.0, top-p 0.95,
/// top-k 64), one setting for every strength; the file declares no
/// `general.sampling.*` keys and the card names no `min_p` or penalties.
pub fn samplingDefaults(effort: Effort) sampling.Options {
    _ = effort;
    return .{ .temperature = 1.0, .top_p = 0.95, .top_k = 64, .min_p = 0, .presence_penalty = 0, .repetition_penalty = 1 };
}

/// The mode's defaults with the caller's overrides applied per option.
pub fn samplingOptions(effort: Effort, overrides: SamplingOverrides) sampling.Options {
    return samplingDefaults(effort).override(overrides);
}

const Builder = struct {
    alloc: std.mem.Allocator,
    limit: usize,
    bytes: std.ArrayList(u8) = .empty,

    fn add(self: *Builder, text: []const u8) Error!void {
        if (text.len > self.limit - self.bytes.items.len) return error.LimitExceeded;
        try self.bytes.appendSlice(self.alloc, text);
    }
};

const Fixture = struct {
    template_sha256: []const u8,
    preserve_thinking: bool,
    prompt_cases: []const struct { name: []const u8, effort: Effort, messages: []const Message, prompt: []const u8, tokens: []const u32 },
};

/// The server filled the synthesized system turn's date line from its
/// clock on capture day; the profile renders none. Owned by `alloc`.
fn withoutDateLine(alloc: std.mem.Allocator, prompt: []const u8) ![]u8 {
    const line = "\nCurrent date: 2026-09-19.";
    const size = std.mem.replacementSize(u8, prompt, line, "");
    const out = try alloc.alloc(u8, size);
    _ = std.mem.replace(u8, prompt, line, "", out);
    return out;
}

test "text prompts match the pinned reference fixtures, BOS prepended, the date line removed" {
    const alloc = std.testing.allocator;
    const fixture = try std.json.parseFromSlice(Fixture, alloc, @embedFile("fixtures/muse_glimmer-text.json"), .{ .ignore_unknown_fields = true });
    defer fixture.deinit();
    try std.testing.expectEqualStrings(template_sha256, fixture.value.template_sha256);
    try std.testing.expect(fixture.value.preserve_thinking);
    try std.testing.expectEqual(@as(usize, 28), fixture.value.prompt_cases.len);
    var dated: usize = 0;
    for (fixture.value.prompt_cases) |case| {
        const undated = try withoutDateLine(alloc, case.prompt);
        defer alloc.free(undated);
        if (undated.len != case.prompt.len) dated += 1;
        const expected = try std.mem.concat(alloc, u8, &.{ bos, undated });
        defer alloc.free(expected);
        const prompt = try render(alloc, case.messages, &.{}, case.effort, .{});
        defer alloc.free(prompt);
        try std.testing.expectEqualStrings(expected, prompt);
        // Every captured stream starts at `<|start|>` (200022): the
        // reference's BOS is implicit there, explicit here.
        try std.testing.expectEqual(@as(u32, 200022), case.tokens[0]);
    }
    // The five conversations without a system message, at four strengths.
    try std.testing.expectEqual(@as(usize, 20), dated);
}

test "a user image renders the image markers before the text" {
    const alloc = std.testing.allocator;
    const refs = [_]profiles.ImageRef{.{ .width_tokens = 3, .height_tokens = 2 }};
    const prompt = try render(alloc, &.{.{ .role = .user, .content = "describe this image", .images = &refs }}, &.{}, .off, .{});
    defer alloc.free(prompt);
    const expected = "<|start|>user<|message|><|image_start|>" ++ "<|patch|>" ** 6 ++ "<|image_end|>describe this image<|eot|><|start|>assistant";
    try std.testing.expect(std.mem.endsWith(u8, prompt, expected));
    // A user cannot type the markers.
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{.{ .role = .user, .content = "a <|patch|> b" }}, &.{}, .off, .{}));
}

test "off renders as low; the strength line names every other level" {
    const alloc = std.testing.allocator;
    const messages = [_]Message{ .{ .role = .system, .content = "Be brief." }, .{ .role = .user, .content = "Hi" } };
    const low = try render(alloc, &messages, &.{}, .low, .{});
    defer alloc.free(low);
    const off = try render(alloc, &messages, &.{}, .off, .{});
    defer alloc.free(off);
    try std.testing.expectEqualStrings(low, off);
    try std.testing.expectEqualStrings("<|begin_of_text|><|start|>system<|message|>Be brief.\n\nReasoning strength: low.\n\n# Valid recipients: \"self\", \"user\".<|eot|><|start|>user<|message|>Hi<|eot|><|start|>assistant", low);
    inline for (.{ .{ .medium, "medium" }, .{ .high, "high" }, .{ .xhigh, "xhigh" } }) |pair| {
        const prompt = try render(alloc, &messages, &.{}, pair[0], .{});
        defer alloc.free(prompt);
        try std.testing.expect(std.mem.indexOf(u8, prompt, "\n\nReasoning strength: " ++ pair[1] ++ ".\n\n") != null);
    }
}

test "history keeps reasoning verbatim as its own message and a developer message is a system turn" {
    const alloc = std.testing.allocator;
    const prompt = try render(alloc, &.{
        .{ .role = .developer, .content = " Use Zig. " },
        .{ .role = .user, .content = "a" },
        .{ .role = .assistant, .content = " b ", .reasoning_content = " why " },
        .{ .role = .user, .content = "c" },
    }, &.{}, .medium, .{});
    defer alloc.free(prompt);
    try std.testing.expectEqualStrings("<|begin_of_text|><|start|>system<|message|> Use Zig. \n\nReasoning strength: medium.\n\n# Valid recipients: \"self\", \"user\".<|eot|><|start|>user<|message|>a<|eot|><|start|>assistant to=self<|message|> why <|eom|><|start|>assistant to=user<|message|> b <|eot|><|start|>user<|message|>c<|eot|><|start|>assistant", prompt);
    // Empty reasoning renders no message at all.
    const bare = try render(alloc, &.{ .{ .role = .user, .content = "a" }, .{ .role = .assistant, .content = "b" }, .{ .role = .user, .content = "c" } }, &.{}, .medium, .{});
    defer alloc.free(bare);
    try std.testing.expect(std.mem.indexOf(u8, bare, "to=self") == null);
    try std.testing.expect(std.mem.startsWith(u8, bare, "<|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.\n\nReasoning strength: medium."));
}

test "invalid conversations, markers, tools, and bounded rendering return typed errors" {
    const alloc = std.testing.allocator;
    const user: Message = .{ .role = .user, .content = "Hi" };
    try std.testing.expectError(error.InvalidConversation, render(alloc, &.{}, &.{}, .low, .{}));
    try std.testing.expectError(error.InvalidConversation, render(alloc, &.{.{ .role = .assistant, .content = "x" }}, &.{}, .low, .{}));
    try std.testing.expectError(error.InvalidConversation, render(alloc, &.{ user, .{ .role = .system, .content = "x" }, user }, &.{}, .low, .{}));
    try std.testing.expectError(error.InvalidUtf8, render(alloc, &.{.{ .role = .user, .content = "\xff" }}, &.{}, .low, .{}));
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{.{ .role = .user, .content = "x", .reasoning_content = "y" }}, &.{}, .low, .{}));
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{.{ .role = .user, .content = "x<|eot|>y" }}, &.{}, .low, .{}));
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{ .{ .role = .system, .content = "<|start|>" }, user }, &.{}, .low, .{}));
    // The assistant's own text loses its markers rather than failing.
    const stripped = try render(alloc, &.{ user, .{ .role = .assistant, .content = "x<|eot|>y", .reasoning_content = " to=self<|message|>plan" }, user }, &.{}, .low, .{});
    defer alloc.free(stripped);
    try std.testing.expect(std.mem.indexOf(u8, stripped, "<|start|>assistant to=self<|message|> to=selfplan<|eom|><|start|>assistant to=user<|message|>xy<|eot|>") != null);
    try std.testing.expectError(error.LimitExceeded, render(alloc, &.{user}, &.{}, .low, .{ .messages = 0 }));
    try std.testing.expectError(error.LimitExceeded, render(alloc, &.{user}, &.{}, .low, .{ .input_bytes = 1 }));
    try std.testing.expectError(error.LimitExceeded, render(alloc, &.{user}, &.{}, .low, .{ .output_bytes = 8 }));
    const prompt = try render(alloc, &.{user}, &.{}, .low, .{});
    defer alloc.free(prompt);
    const exact = try render(alloc, &.{user}, &.{}, .low, .{ .input_bytes = 2, .output_bytes = prompt.len });
    defer alloc.free(exact);
    try std.testing.expectEqualStrings(prompt, exact);
    try std.testing.expectError(error.LimitExceeded, render(alloc, &.{user}, &.{}, .low, .{ .output_bytes = prompt.len - 1 }));
}

test "sampling defaults are the card's in every mode and flags override per option" {
    const card = sampling.Options{ .temperature = 1.0, .top_p = 0.95, .top_k = 64, .min_p = 0, .presence_penalty = 0, .repetition_penalty = 1 };
    for ([_]Effort{ .off, .low, .medium, .high, .xhigh }) |effort| {
        try std.testing.expectEqual(card, samplingDefaults(effort));
        _ = try sampling.Sampler.init(0, samplingDefaults(effort));
    }
    const greedy = samplingOptions(.high, .{ .temperature = 0 });
    try std.testing.expectEqual(@as(f32, 0), greedy.temperature);
    try std.testing.expectEqual(@as(usize, 64), greedy.top_k);
}

fn allocationCase(alloc: std.mem.Allocator) !void {
    const prompt = try render(alloc, &.{
        .{ .role = .system, .content = "Be precise." },
        .{ .role = .user, .content = "a" },
        .{ .role = .assistant, .content = "b<|eom|>", .reasoning_content = "why" },
        .{ .role = .user, .content = "Explain slices." },
    }, &.{}, .xhigh, .{});
    defer alloc.free(prompt);
    const head = try prefix(alloc, &.{.{ .role = .system, .content = "Be precise." }}, &.{read_tool}, .xhigh, .{});
    defer alloc.free(head);
    const call: profiles.ToolCall = .{ .id = 1, .name = "read_file", .arguments = "{\"path\":\"a\",\"n\":[1,{\"b\":true}]}" };
    const with_tools = try render(alloc, &.{
        .{ .role = .user, .content = "go" },
        .{ .role = .assistant, .content = "", .reasoning_content = "Plan.", .tool_calls = &.{call} },
        .{ .role = .tool, .content = "body", .tool_call_id = 1 },
    }, &.{read_tool}, .medium, .{});
    defer alloc.free(with_tools);
    if (try parseTool(alloc, "<atem:function_calls>\n<atem:invoke name=\"f\">\n<atem:parameter name=\"a\">x</atem:parameter>\n<atem:parameter name=\"b\">[1, {\"c\": null}]</atem:parameter>\n</atem:invoke>\n</atem:function_calls>")) |parsed| {
        alloc.free(parsed.name);
        alloc.free(parsed.arguments);
    } else return error.UnexpectedNull;
}

const read_tool: profiles.ToolDefinition = .{
    .name = "read_file",
    .description = "Read a file.",
    .parameters = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"offset\":{\"type\":\"integer\"}},\"required\":[\"path\"]}",
};

test "prompt rendering cleans up every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationCase, .{});
}

test "the prefix is a byte prefix of every rendering that starts with its system messages" {
    const alloc = std.testing.allocator;
    const system: Message = .{ .role = .system, .content = "You are terse." };
    const developer: Message = .{ .role = .developer, .content = "Use Zig." };
    const user: Message = .{ .role = .user, .content = "hello" };
    inline for (.{ .off, .high }) |effort| {
        for ([_][]const Message{ &.{ system, developer, user }, &.{ system, user }, &.{user} }) |messages| {
            for ([_][]const profiles.ToolDefinition{ &.{}, &.{read_tool} }) |tools| {
                const head = try prefix(alloc, messages[0 .. messages.len - 1], tools, effort, .{});
                defer alloc.free(head);
                const full = try render(alloc, messages, tools, effort, .{});
                defer alloc.free(full);
                try std.testing.expect(std.mem.startsWith(u8, full, head));
                try std.testing.expect(std.mem.startsWith(u8, full[head.len..], "<|start|>user<|message|>"));
            }
        }
    }
}

test "native tool prompts match the pinned reference fixtures, BOS prepended, the date line removed" {
    const ToolFixture = struct {
        template_sha256: []const u8,
        preserve_thinking: bool,
        cases: []const struct {
            name: []const u8,
            effort: Effort,
            messages: []const Message,
            tools: []const profiles.ToolDefinition,
            prompt: []const u8,
        },
    };
    const alloc = std.testing.allocator;
    const fixture = try std.json.parseFromSlice(ToolFixture, alloc, @embedFile("fixtures/muse_glimmer-tools.json"), .{ .ignore_unknown_fields = true });
    defer fixture.deinit();
    try std.testing.expectEqualStrings(template_sha256, fixture.value.template_sha256);
    try std.testing.expect(fixture.value.preserve_thinking);
    try std.testing.expectEqual(@as(usize, 52), fixture.value.cases.len);
    for (fixture.value.cases) |case| {
        const undated = try withoutDateLine(alloc, case.prompt);
        defer alloc.free(undated);
        const expected = try std.mem.concat(alloc, u8, &.{ bos, undated });
        defer alloc.free(expected);
        const prompt = try render(alloc, case.messages, case.tools, case.effort, .{});
        defer alloc.free(prompt);
        try std.testing.expectEqualStrings(expected, prompt);
    }
}

test "parseTool round-trips a rendered call and keeps every value's type" {
    const alloc = std.testing.allocator;
    const arguments = "{\"query\":\"a\\\"b\\nc\",\"limit\":3,\"ratio\":1.5,\"regex\":true,\"nothing\":null,\"paths\":[\"x\",\"y\"],\"options\":{\"case_sensitive\":false},\"path\":\"a.zig\",\"padded\":\" spaced \"}";
    const messages = [_]Message{
        .{ .role = .user, .content = "search" },
        .{ .role = .assistant, .content = "", .tool_calls = &.{.{ .id = 1, .name = "search", .arguments = arguments }} },
        .{ .role = .tool, .content = "none", .tool_call_id = 1 },
        .{ .role = .user, .content = "ok" },
    };
    const prompt = try render(alloc, &messages, &.{}, .low, .{});
    defer alloc.free(prompt);
    const open_marker = "<|start|>assistant to=search<|message|>";
    const open = std.mem.indexOf(u8, prompt, open_marker).? + open_marker.len;
    const close = std.mem.indexOfPos(u8, prompt, open, "<|eot|>").?;
    const parsed = (try parseTool(alloc, prompt[open..close])).?;
    defer {
        alloc.free(parsed.name);
        alloc.free(parsed.arguments);
    }
    try std.testing.expectEqualStrings("search", parsed.name);
    // Strings are verbatim on the wire (a quote and a newline survive; a
    // leading space is kept), JSON values keep their types.
    try std.testing.expectEqualStrings("{\"query\":\"a\\\"b\\nc\",\"limit\":3,\"ratio\":1.5,\"regex\":true,\"nothing\":null,\"paths\":[\"x\",\"y\"],\"options\":{\"case_sensitive\":false},\"path\":\"a.zig\",\"padded\":\" spaced \"}", parsed.arguments);
}

test "a string value's newlines at either end survive the round trip" {
    const alloc = std.testing.allocator;
    // The reference's grammar reads a value as everything up to the closing
    // tag, so a file's trailing newline and a leading blank line are content.
    const cases = [_][]const u8{
        "{\"path\":\"a.txt\",\"content\":\"one\\ntwo\\n\"}",
        "{\"path\":\"a.txt\",\"content\":\"\\nafter a blank line\\n\\n\"}",
        "{\"path\":\"a.txt\",\"content\":\"\\n\"}",
    };
    for (cases) |arguments| {
        const messages = [_]Message{
            .{ .role = .user, .content = "write" },
            .{ .role = .assistant, .content = "", .tool_calls = &.{.{ .id = 1, .name = "write_file", .arguments = arguments }} },
            .{ .role = .tool, .content = "ok", .tool_call_id = 1 },
        };
        const prompt = try render(alloc, &messages, &.{}, .low, .{});
        defer alloc.free(prompt);
        const open_marker = "<|start|>assistant to=write_file<|message|>";
        const open = std.mem.indexOf(u8, prompt, open_marker).? + open_marker.len;
        const close = std.mem.indexOfPos(u8, prompt, open, "<|eot|>").?;
        const parsed = (try parseTool(alloc, prompt[open..close])).?;
        defer {
            alloc.free(parsed.name);
            alloc.free(parsed.arguments);
        }
        try std.testing.expectEqualStrings(arguments, parsed.arguments);
    }
}

test "parseTool accepts the block's whitespace and refuses malformed, truncated, or multiple invokes" {
    const alloc = std.testing.allocator;
    const spaced = (try parseTool(alloc, "\n<atem:function_calls>\n\n<atem:invoke name=\"bash\">\n  <atem:parameter name=\"command\">ls -la</atem:parameter>\n\n</atem:invoke>\n</atem:function_calls>\n\n")).?;
    defer {
        alloc.free(spaced.name);
        alloc.free(spaced.arguments);
    }
    try std.testing.expectEqualStrings("{\"command\":\"ls -la\"}", spaced.arguments);
    const empty = (try parseTool(alloc, "<atem:function_calls>\n<atem:invoke name=\"bash\">\n</atem:invoke>\n</atem:function_calls>")).?;
    defer {
        alloc.free(empty.name);
        alloc.free(empty.arguments);
    }
    try std.testing.expectEqualStrings("bash", empty.name);
    try std.testing.expectEqualStrings("{}", empty.arguments);
    for ([_][]const u8{
        "no call here",
        "<atem:function_calls>\n<atem:invoke name=\"\">\n</atem:invoke>\n</atem:function_calls>", // empty name
        "<atem:function_calls>\n<atem:invoke name=\"a b\">\n</atem:invoke>\n</atem:function_calls>", // whitespace in the name
        "<atem:function_calls>\n<atem:invoke name=\"f\">\n<atem:parameter name=\"a\">1</atem:parameter>\n", // truncated
        "<atem:function_calls>\n<atem:invoke name=\"f\">\n<atem:parameter name=\"a\">open\n</atem:invoke>\n</atem:function_calls>", // unterminated parameter
        "<atem:function_calls>\n<atem:invoke name=\"f\">\n</atem:invoke>\n</atem:function_calls>x", // trailing text
        "<atem:function_calls>\n<atem:invoke name=\"f\">\n<atem:parameter name=\"\">1</atem:parameter>\n</atem:invoke>\n</atem:function_calls>", // empty key
        "<atem:function_calls>\n<atem:invoke name=\"f\">\n</atem:invoke>\n<atem:invoke name=\"g\">\n</atem:invoke>\n</atem:function_calls>", // two invokes
        "<atem:function_calls>\n<atem:invoke name=\"f\">\nstray</atem:invoke>\n</atem:function_calls>", // text where a parameter belongs
    }) |body| {
        try std.testing.expect((try parseTool(alloc, body)) == null);
    }
}

test "the channel decoder delivers two calls in one turn and releases an unfinished one" {
    const alloc = std.testing.allocator;
    const call_markers: profiles.stream.Markers = .{ .open = 100, .close = 101, .channel = .{ .start = 100, .message = 102, .eom = 101, .parse = parseTool } };
    const body1 = "<atem:function_calls>\n<atem:invoke name=\"read_file\">\n<atem:parameter name=\"path\">a.zig</atem:parameter>\n<atem:parameter name=\"offset\">3</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
    const body2 = "<atem:function_calls>\n<atem:invoke name=\"bash\">\n<atem:parameter name=\"command\">ls</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
    var sink = Sink.init(alloc);
    defer sink.deinit();
    var d = profiles.stream.Decoder.init(alloc, call_markers, true);
    defer d.deinit();
    try d.feed(7, " to=self", &sink);
    try d.feed(102, "<|message|>", &sink);
    try d.feed(7, "plan", &sink);
    try d.feed(101, "<|eom|>", &sink);
    try d.feed(100, "<|start|>", &sink);
    try d.feed(7, "assistant to=read_file", &sink);
    try d.feed(102, "<|message|>", &sink);
    // The body is ordinary text; a token can split it at any byte.
    for (body1) |byte| try d.feed(7, &.{byte}, &sink);
    try d.feed(101, "<|eom|>", &sink);
    try d.feed(100, "<|start|>", &sink);
    try d.feed(7, "assistant to=bash", &sink);
    try d.feed(102, "<|message|>", &sink);
    try d.feed(7, body2, &sink);
    // `<|eot|>` is a stop token: the loop ends the turn with EOS and the
    // second body completes there.
    try d.end(.{ .stop = .eos, .timing = .{ .prompt_tokens = 1, .generated_tokens = 1 } }, &sink);
    try std.testing.expectEqualStrings("plan", sink.thinking.written());
    try std.testing.expectEqualStrings("", sink.answer.written());
    try std.testing.expectEqual(@as(usize, 2), sink.calls.items.len);
    try std.testing.expectEqualStrings("read_file", sink.calls.items[0].name);
    try std.testing.expectEqualStrings("{\"path\":\"a.zig\",\"offset\":3}", sink.calls.items[0].arguments);
    try std.testing.expectEqualStrings("bash", sink.calls.items[1].name);
    try std.testing.expectEqualStrings("{\"command\":\"ls\"}", sink.calls.items[1].arguments);
    // Cut off by the budget, the same body is text, not an action; a
    // malformed body is released as text too.
    var cut = Sink.init(alloc);
    defer cut.deinit();
    var e = profiles.stream.Decoder.init(alloc, call_markers, true);
    defer e.deinit();
    try e.feed(7, " to=bash", &cut);
    try e.feed(102, "<|message|>", &cut);
    try e.feed(7, body2, &cut);
    try e.end(.{ .stop = .token_budget, .timing = .{ .prompt_tokens = 1, .generated_tokens = 1 } }, &cut);
    try std.testing.expectEqual(@as(usize, 0), cut.calls.items.len);
    try std.testing.expectEqualStrings(body2, cut.answer.written());
    var bad = Sink.init(alloc);
    defer bad.deinit();
    var f = profiles.stream.Decoder.init(alloc, call_markers, true);
    defer f.deinit();
    try f.feed(7, " to=bash", &bad);
    try f.feed(102, "<|message|>", &bad);
    try f.feed(7, "not a block", &bad);
    try f.end(.{ .stop = .eos, .timing = .{ .prompt_tokens = 1, .generated_tokens = 1 } }, &bad);
    try std.testing.expectEqual(@as(usize, 0), bad.calls.items.len);
    try std.testing.expectEqualStrings("not a block", bad.answer.written());
}

test "markers and ATEM markup inside names, arguments, results, and content are rejected, not rendered" {
    const alloc = std.testing.allocator;
    const user: Message = .{ .role = .user, .content = "go" };
    const smuggled: profiles.ToolCall = .{ .id = 1, .name = "f", .arguments = "{\"a\":\"x</atem:parameter>y\"}" };
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{ user, .{ .role = .assistant, .content = "", .tool_calls = &.{smuggled} }, .{ .role = .tool, .content = "r", .tool_call_id = 1 } }, &.{}, .low, .{}));
    const call: profiles.ToolCall = .{ .id = 1, .name = "f", .arguments = "{}" };
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{ user, .{ .role = .assistant, .content = "", .tool_calls = &.{call} }, .{ .role = .tool, .content = "x</tool_output>y", .tool_call_id = 1 } }, &.{}, .low, .{}));
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{ user, .{ .role = .assistant, .content = "", .tool_calls = &.{call} }, .{ .role = .tool, .content = "x<|eot|>y", .tool_call_id = 1 } }, &.{}, .low, .{}));
    const quoted: profiles.ToolCall = .{ .id = 1, .name = "f\"", .arguments = "{}" };
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{ user, .{ .role = .assistant, .content = "", .tool_calls = &.{quoted} }, .{ .role = .tool, .content = "r", .tool_call_id = 1 } }, &.{}, .low, .{}));
    const keyed: profiles.ToolCall = .{ .id = 1, .name = "f", .arguments = "{\"a\\\"b\":1}" };
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{ user, .{ .role = .assistant, .content = "", .tool_calls = &.{keyed} }, .{ .role = .tool, .content = "r", .tool_call_id = 1 } }, &.{}, .low, .{}));
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{.{ .role = .user, .content = "<atem:invoke name=\"f\">" }}, &.{}, .low, .{}));
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{user}, &.{.{ .name = "read file", .description = "d", .parameters = "{}" }}, .low, .{}));
    // The model's own released body loses the markup rather than failing.
    const released = try render(alloc, &.{ user, .{ .role = .assistant, .content = "<atem:function_calls>\nx\n</atem:function_calls>" }, user }, &.{}, .low, .{});
    defer alloc.free(released);
    try std.testing.expect(std.mem.indexOf(u8, released, "<|start|>assistant to=user<|message|>\nx\n<|eot|>") != null);
    // A namespaced tool changes the recipients line, once per namespace.
    const namespaced = try render(alloc, &.{user}, &.{ .{ .name = "fs.read", .description = "d", .parameters = "{}" }, .{ .name = "fs.write", .description = "d", .parameters = "{}" }, .{ .name = "bash", .description = "d", .parameters = "{}" } }, .low, .{});
    defer alloc.free(namespaced);
    try std.testing.expect(std.mem.indexOf(u8, namespaced, "# Valid recipients: \"self\", \"fs.*\", \"bash.*\", \"user\".<|eot|>") != null);
    try std.testing.expect(std.mem.indexOf(u8, namespaced, "// Tool metadata\n{\"name\": \"fs\", \"description\": \"\"}\n{\"name\": \"bash\", \"description\": \"\"}\n// Function schemas\n{\"name\": \"fs.read\", \"description\": \"d\", \"parameters\": {}}\n{\"name\": \"fs.write\"") != null);
}

/// A sink that retains every event, for the decoder tests.
const Sink = struct {
    alloc: std.mem.Allocator,
    thinking: std.Io.Writer.Allocating,
    answer: std.Io.Writer.Allocating,
    calls: std.ArrayList(profiles.ToolCall) = .empty,
    stops: usize = 0,

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
            .stop => self.stops += 1,
        }
    }
};

/// The fake ids the decoder tests use; production ids differ deliberately.
const test_channel: profiles.stream.Markers = .{ .open = 100, .close = 101, .channel = .{ .start = 100, .message = 102, .eom = 101 } };

test "the channel decoder routes the first header from text and later ones from the start token" {
    const alloc = std.testing.allocator;
    // The header text can split at every byte: ` to=self` then a body,
    // `<|eom|>`, `<|start|>assistant to=user<|message|>`, the answer.
    const header = " to=self";
    for (0..header.len + 1) |split| {
        var sink = Sink.init(alloc);
        defer sink.deinit();
        var d = profiles.stream.Decoder.init(alloc, test_channel, true);
        defer d.deinit();
        try d.feed(7, header[0..split], &sink);
        try d.feed(7, header[split..], &sink);
        try d.feed(102, "<|message|>", &sink);
        try d.feed(7, "plan é", &sink);
        try d.feed(101, "<|eom|>", &sink);
        try d.feed(100, "<|start|>", &sink);
        try d.feed(7, "assistant", &sink);
        try d.feed(7, " to=user", &sink);
        try d.feed(102, "<|message|>", &sink);
        try d.feed(7, "reply", &sink);
        try d.end(.{ .stop = .eos, .timing = .{ .prompt_tokens = 1, .generated_tokens = 1 } }, &sink);
        try std.testing.expectEqualStrings("plan é", sink.thinking.written());
        try std.testing.expectEqualStrings("reply", sink.answer.written());
        try std.testing.expectEqual(@as(usize, 1), sink.stops);
    }
}

test "a bare header, a truncated header, and an overlong header hide nothing" {
    const alloc = std.testing.allocator;
    // `<|message|>` right after the generation prompt: the answer.
    var sink = Sink.init(alloc);
    defer sink.deinit();
    var d = profiles.stream.Decoder.init(alloc, test_channel, true);
    defer d.deinit();
    try d.feed(102, "<|message|>", &sink);
    try d.feed(7, "direct", &sink);
    try d.finish(&sink);
    try std.testing.expectEqualStrings("direct", sink.answer.written());
    try std.testing.expectEqualStrings("", sink.thinking.written());
    // A header cut off by the budget is released as answer text.
    var cut = Sink.init(alloc);
    defer cut.deinit();
    var e = profiles.stream.Decoder.init(alloc, test_channel, true);
    defer e.deinit();
    try e.feed(7, " to=se", &cut);
    try e.finish(&cut);
    try std.testing.expectEqualStrings(" to=se", cut.answer.written());
    // Prose where a header belongs stops being collected past the bound.
    var long = Sink.init(alloc);
    defer long.deinit();
    var f = profiles.stream.Decoder.init(alloc, test_channel, true);
    defer f.deinit();
    const prose = "x" ** (profiles.stream.header_limit + 5);
    try f.feed(7, prose, &long);
    try f.feed(7, "more", &long);
    try f.finish(&long);
    try std.testing.expectEqualStrings(prose ++ "more", long.answer.written());
    // A recipient the profile cannot parse (no tool parser) is answer text.
    var other = Sink.init(alloc);
    defer other.deinit();
    var g = profiles.stream.Decoder.init(alloc, test_channel, true);
    defer g.deinit();
    try g.feed(7, " to=read_file", &other);
    try g.feed(102, "<|message|>", &other);
    try g.feed(7, "body", &other);
    try g.finish(&other);
    try std.testing.expectEqualStrings("body", other.answer.written());
}
