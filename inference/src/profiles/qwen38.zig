//! Bounded prompt profile for the pinned Qwen3.8 artifact.
//! Adapted from its Qwen/Unsloth GGUF chat template (Apache-2.0; see THIRD_PARTY_NOTICES.md).
//! This Zig implementation supports separate assistant reasoning and preserves
//! it in history, renders a native tools block, and renders assistant calls and
//! tool results in the template's own syntax. Multimodal content and assistant
//! prefill are absent. It owns no tokenizer or model equations. See
//! docs/reference/prompt-profile.md.
//!
//! The profile also carries the model's official per-mode sampling defaults
//! (`samplingDefaults`): they belong to the checkpoint's recommended usage,
//! not to the shared sampler, which stays a policy engine with neutral
//! defaults.
const std = @import("std");
const sampling = @import("../sampling/root.zig");

/// Exact GGUF template this profile implements; future loading must check it
/// before selecting this profile instead of guessing from architecture alone.
pub const template_sha256 = "12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce";
/// Other template revisions proven to render every fixture case identically
/// (`scripts/profile-alias-check.py`); none so far.
pub const template_aliases = [_][]const u8{};

/// The conversation types are the registry's (`profiles/root.zig`), shared
/// with every profile; re-exported so this module reads on its own.
const profiles = @import("root.zig");
pub const Role = profiles.Role;
pub const Effort = profiles.Effort;
pub const Message = profiles.Message;
pub const Limits = profiles.Limits;
pub const Error = profiles.Error;

/// The tokens that end a turn under this template: `<|im_end|>` (248046,
/// the file's `eos_token_id`) and `<|endoftext|>` (248044, its
/// `bos_token_id`; the model emits it at the end of some answers).
pub const stop_tokens = [_][]const u8{ "<|im_end|>", "<|endoftext|>" };

/// How the model delimits its reasoning: the prompt opens `<think>` and the
/// model closes it with `</think>`.
pub const reasoning: profiles.Reasoning = .{ .open = "<think>", .close = "</think>" };
pub const stream_markers: profiles.StreamMarkers = .{
    .open = "<think>",
    .close = "</think>",
    .open_suffix = "",
    .tool_open = "<tool_call>",
    .tool_close = "</tool_call>",
    .tool_parse = parseTool,
};

/// The tools-block header and its format explanation, and the framing around
/// one rendered call. These strings are the pinned artifact's, byte for byte
/// (`fixtures/qwen38-tools.json`); no other module names this wire syntax.
const tools_header = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const tools_format = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// Returns an owned UTF-8 prompt; caller frees it with the supplied allocator.
/// All message strings are borrowed only for this call. `profiles.validate`
/// enforces the shared conversation rules (completion-ready ending, role
/// order, bounds, tool history). Assistant reasoning is supplied separately,
/// never extracted from content. Leading/trailing ASCII whitespace follows the
/// pinned reference, as does the tools block, the rendered calls, and the
/// folding of consecutive tool results into one user turn.
pub fn render(alloc: std.mem.Allocator, messages: []const Message, tools: []const profiles.ToolDefinition, effort: Effort, limits: Limits) Error![]u8 {
    try profiles.validate(alloc, messages, tools, limits);
    const prefix_count = profiles.leadingSystemCount(messages);
    const instruction = instructionFor(effort);
    const merged = try mergeSystem(alloc, messages[0..prefix_count]);
    defer alloc.free(merged);

    var builder: Builder = .{ .alloc = alloc, .limit = limits.output_bytes };
    errdefer builder.bytes.deinit(alloc);
    try renderSystem(&builder, alloc, tools, instruction, merged);

    var i = prefix_count;
    while (i < messages.len) : (i += 1) {
        const message = messages[i];
        switch (message.role) {
            // `validate` rejects a system or developer message after the prefix.
            .system, .developer => unreachable,
            .user => {
                try builder.add("<|im_start|>user\n");
                // Qwen3-VL markers, one run per image, before the text: the
                // vision boundary and `count` image placeholders the engine's
                // spans overwrite with projector rows.
                for (message.images) |image| {
                    try builder.add("<|vision_start|>");
                    var n: usize = 0;
                    const count = @as(usize, image.width_tokens) * image.height_tokens;
                    while (n < count) : (n += 1) try builder.add("<|image_pad|>");
                    try builder.add("<|vision_end|>");
                }
                try builder.add(trim(message.content));
                try builder.add("<|im_end|>\n");
            },
            .assistant => {
                // The model's own past text may carry marker text (a call
                // released as text when its close never came): re-encoding
                // it would turn text into control tokens, so the markers
                // are removed.
                const own_reasoning = try stripMarkers(alloc, message.reasoning_content);
                defer alloc.free(own_reasoning);
                const own_content = try stripMarkers(alloc, message.content);
                defer alloc.free(own_content);
                try builder.add("<|im_start|>assistant\n<think>\n");
                try builder.add(trim(own_reasoning));
                try builder.add("\n</think>\n\n");
                const content = trim(own_content);
                try builder.add(content);
                for (message.tool_calls, 0..) |call, c| {
                    if (c == 0) {
                        if (content.len != 0) try builder.add("\n\n");
                    } else try builder.add("\n");
                    try renderCall(&builder, alloc, call);
                }
                try builder.add("<|im_end|>\n");
            },
            // Consecutive tool results are one user turn; the template opens it
            // on the first and closes it on the last of the run.
            .tool => {
                if (i > 0 and messages[i - 1].role != .tool) try builder.add("<|im_start|>user");
                try builder.add("\n<tool_response>\n");
                try builder.add(trim(message.content));
                try builder.add("\n</tool_response>");
                if (i + 1 >= messages.len or messages[i + 1].role != .tool) try builder.add("<|im_end|>\n");
            },
        }
    }
    try builder.add("<|im_start|>assistant\n<think>\n");
    if (effort == .off) try builder.add("\n</think>\n\n");
    return builder.bytes.toOwnedSlice(alloc);
}

/// The bytes every rendering of a conversation with these leading system
/// messages and tools starts with: the system block alone, without the
/// generation prompt `render` ends with. What a session primes with before
/// its first user message; pinned as a byte prefix of `render` by a test.
pub fn prefix(alloc: std.mem.Allocator, messages: []const Message, tools: []const profiles.ToolDefinition, effort: Effort, limits: Limits) Error![]u8 {
    const prefix_count = profiles.leadingSystemCount(messages);
    const merged = try mergeSystem(alloc, messages[0..prefix_count]);
    defer alloc.free(merged);
    var builder: Builder = .{ .alloc = alloc, .limit = limits.output_bytes };
    errdefer builder.bytes.deinit(alloc);
    try renderSystem(&builder, alloc, tools, instructionFor(effort), merged);
    return builder.bytes.toOwnedSlice(alloc);
}

/// The leading system block. With tool definitions the tools text comes first
/// and the merged system messages follow it; without them the layout is the
/// text-only one. Both are the pinned template's.
fn renderSystem(builder: *Builder, alloc: std.mem.Allocator, tools: []const profiles.ToolDefinition, instruction: []const u8, merged: []const u8) Error!void {
    if (tools.len != 0) {
        try builder.add("<|im_start|>system\n");
        if (instruction.len != 0) {
            try builder.add(instruction);
            try builder.add("\n\n");
        }
        try builder.add(tools_header);
        for (tools) |tool| {
            try builder.add("\n");
            try addToolDefinition(builder, alloc, tool);
        }
        try builder.add("\n</tools>");
        try builder.add(tools_format);
        if (merged.len != 0) {
            try builder.add("\n\n");
            try builder.add(merged);
        }
        try builder.add("<|im_end|>\n");
        return;
    }
    if (merged.len == 0 and instruction.len == 0) return;
    try builder.add("<|im_start|>system\n");
    try builder.add(instruction);
    if (merged.len != 0) {
        if (instruction.len != 0) try builder.add("\n\n");
        try builder.add(merged);
    }
    try builder.add("<|im_end|>\n");
}

/// The reasoning-effort instruction the template inserts. `medium` has none;
/// `off` (thinking disabled) never reaches this text; the template resolves
/// `high` to `xhigh` before choosing the line (the fixture's `*_high` cases).
fn instructionFor(effort: Effort) []const u8 {
    return switch (effort) {
        .off, .medium => "",
        .low => "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
        .high, .xhigh => "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
    };
}

/// Every delimiter the template spells inside a turn.
const markers = [_][]const u8{ "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<|im_start|>", "<|im_end|>" };

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

/// The leading system/developer messages merged as the template does: trimmed,
/// empty parts dropped, joined with `\n`.
fn mergeSystem(alloc: std.mem.Allocator, leading: []const Message) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    for (leading) |message| {
        const content = trim(message.content);
        if (content.len == 0) continue;
        if (out.items.len != 0) try out.append(alloc, '\n');
        try out.appendSlice(alloc, content);
    }
    return out.toOwnedSlice(alloc);
}

/// One tool definition as the reference serializes it: the OpenAI-shaped
/// object with the schema embedded verbatim as a parsed value, in the
/// reference's `tojson` style (a space after `:` and between items).
fn addToolDefinition(builder: *Builder, alloc: std.mem.Allocator, tool: profiles.ToolDefinition) Error!void {
    var w: std.Io.Writer.Allocating = .init(alloc);
    defer w.deinit();
    writeJsonRaw(&w.writer, "{\"type\": \"function\", \"function\": {\"name\": ") catch return error.OutOfMemory;
    writeJsonString(&w.writer, tool.name) catch return error.OutOfMemory;
    writeJsonRaw(&w.writer, ", \"description\": ") catch return error.OutOfMemory;
    writeJsonString(&w.writer, tool.description) catch return error.OutOfMemory;
    writeJsonRaw(&w.writer, ", \"parameters\": ") catch return error.OutOfMemory;
    // `validate` already proved this is a JSON object.
    const parsed = std.json.parseFromSlice(std.json.Value, alloc, tool.parameters, .{}) catch |err| return if (err == error.OutOfMemory) error.OutOfMemory else error.InvalidConversation;
    defer parsed.deinit();
    writeJsonValue(&w.writer, parsed.value) catch return error.OutOfMemory;
    writeJsonRaw(&w.writer, "}}") catch return error.OutOfMemory;
    try builder.add(w.written());
}

/// One assistant call in the template's native syntax. String argument values
/// are literal; every other value is JSON, exactly as history rendering does.
fn renderCall(builder: *Builder, alloc: std.mem.Allocator, call: profiles.ToolCall) Error!void {
    try builder.add("<tool_call>\n<function=");
    try builder.add(call.name);
    try builder.add(">\n");
    // `validate` already proved these arguments are a JSON object.
    const parsed = std.json.parseFromSlice(std.json.Value, alloc, call.arguments, .{}) catch |err| return if (err == error.OutOfMemory) error.OutOfMemory else error.InvalidConversation;
    defer parsed.deinit();
    var it = parsed.value.object.iterator();
    while (it.next()) |entry| {
        try builder.add("<parameter=");
        try builder.add(entry.key_ptr.*);
        try builder.add(">\n");
        const value = entry.value_ptr.*;
        if (value == .string) {
            try builder.add(value.string);
        } else {
            var w: std.Io.Writer.Allocating = .init(alloc);
            defer w.deinit();
            writeJsonValue(&w.writer, value) catch return error.OutOfMemory;
            try builder.add(w.written());
        }
        try builder.add("\n</parameter>\n");
    }
    try builder.add("</function>\n</tool_call>");
}

fn writeJsonRaw(w: *std.Io.Writer, text: []const u8) std.Io.Writer.Error!void {
    try w.writeAll(text);
}

/// The reference's `tojson`: no newlines, a space after `:` and between items,
/// keys in insertion order, non-ASCII left literal.
fn writeJsonValue(w: *std.Io.Writer, value: std.json.Value) std.Io.Writer.Error!void {
    switch (value) {
        .null => try w.writeAll("null"),
        .bool => |b| try w.writeAll(if (b) "true" else "false"),
        .integer => |n| try w.print("{d}", .{n}),
        .float => |f| try w.print("{d}", .{f}),
        .number_string => |s| try w.writeAll(s),
        .string => |s| try writeJsonString(w, s),
        .array => |array| {
            try w.writeByte('[');
            for (array.items, 0..) |item, i| {
                if (i != 0) try w.writeAll(", ");
                try writeJsonValue(w, item);
            }
            try w.writeByte(']');
        },
        .object => |object| {
            try w.writeByte('{');
            var it = object.iterator();
            var first = true;
            while (it.next()) |entry| {
                if (!first) try w.writeAll(", ");
                first = false;
                try writeJsonString(w, entry.key_ptr.*);
                try w.writeAll(": ");
                try writeJsonValue(w, entry.value_ptr.*);
            }
            try w.writeByte('}');
        },
    }
}

fn writeJsonString(w: *std.Io.Writer, text: []const u8) std.Io.Writer.Error!void {
    try w.writeByte('"');
    for (text) |byte| switch (byte) {
        '"' => try w.writeAll("\\\""),
        '\\' => try w.writeAll("\\\\"),
        '\n' => try w.writeAll("\\n"),
        '\r' => try w.writeAll("\\r"),
        '\t' => try w.writeAll("\\t"),
        0x08 => try w.writeAll("\\b"),
        0x0c => try w.writeAll("\\f"),
        else => if (byte < 0x20) try w.print("\\u{x:0>4}", .{byte}) else try w.writeByte(byte),
    };
    try w.writeByte('"');
}

/// Parses one native call body — the text between the `<tool_call>` and
/// `</tool_call>` control tokens — into an owned `ToolCall`, or null when the
/// body is not one complete, well-formed call. This is the inverse of
/// `renderCall`: a value that parses as JSON keeps its type, anything else is
/// a literal string, and only the template's delimiters around a value are
/// removed (`delimited`), so content keeps its own trailing newline.
pub fn parseTool(alloc: std.mem.Allocator, body: []const u8) std.mem.Allocator.Error!?profiles.ToolCall {
    const function_marker = "<function=";
    const parameter_marker = "<parameter=";
    const parameter_close = "</parameter>";
    const at = std.mem.indexOf(u8, body, function_marker) orelse return null;
    const start = at + function_marker.len;
    const end = std.mem.indexOfScalarPos(u8, body, start, '>') orelse return null;
    const name = std.mem.trim(u8, body[start..end], " \t\r\n");
    if (name.len == 0) return null;
    const name_owned = try alloc.dupe(u8, name);
    errdefer alloc.free(name_owned);

    var arguments: std.Io.Writer.Allocating = .init(alloc);
    errdefer arguments.deinit();
    var json: std.json.Stringify = .{ .writer = &arguments.writer };
    json.beginObject() catch return error.OutOfMemory;
    var cursor = end + 1;
    while (std.mem.indexOfPos(u8, body, cursor, parameter_marker)) |pstart| {
        const key_start = pstart + parameter_marker.len;
        const key_end = std.mem.indexOfScalarPos(u8, body, key_start, '>') orelse break;
        const key = std.mem.trim(u8, body[key_start..key_end], " \t\r\n");
        const value_start = key_end + 1;
        const value_end = std.mem.indexOfPos(u8, body, value_start, parameter_close) orelse break;
        if (key.len > 0) {
            json.objectField(key) catch return error.OutOfMemory;
            try writeArgumentValue(&json, alloc, delimited(body[value_start..value_end]));
        }
        cursor = value_end + parameter_close.len;
    }
    json.endObject() catch return error.OutOfMemory;
    const arguments_owned = try arguments.toOwnedSlice();
    errdefer alloc.free(arguments_owned);
    return .{ .id = 0, .name = name_owned, .arguments = arguments_owned };
}

/// A parameter value on the wire: JSON when it parses as JSON, a literal
/// string otherwise — the template's own rule.
/// A parameter value with the template's delimiters removed — one newline
/// after `>` and one before `</parameter>`, the reference's grammar — and
/// nothing else: a value keeps its own leading spaces and trailing newline.
fn delimited(raw: []const u8) []const u8 {
    var text = raw;
    if (std.mem.startsWith(u8, text, "\n")) text = text[1..];
    if (std.mem.endsWith(u8, text, "\n")) text = text[0 .. text.len - 1];
    return text;
}

/// The JSON typing looks at the whitespace-trimmed text, so ` 20 ` is the
/// number 20; a value that is not JSON is written as the string it is.
fn writeArgumentValue(json: *std.json.Stringify, alloc: std.mem.Allocator, raw: []const u8) std.mem.Allocator.Error!void {
    const typed = std.mem.trim(u8, raw, " \t\r\n");
    if (typed.len > 0) {
        if (std.json.parseFromSlice(std.json.Value, alloc, typed, .{})) |parsed| {
            defer parsed.deinit();
            json.write(parsed.value) catch return error.OutOfMemory;
            return;
        } else |_| {}
    }
    json.write(raw) catch return error.OutOfMemory;
}

pub const SamplingOverrides = profiles.SamplingOverrides;

/// The official Qwen3.8 sampler settings per reasoning mode, from the Unsloth
/// Qwen3.8 guide (https://unsloth.ai/docs/models/qwen3.8.md, read 2026-09-08):
/// thinking mode (`low`, `medium`, `high`, `xhigh`) temperature 1.0, top-p 0.95,
/// top-k 20; instruct mode (`off`) temperature 0.7, top-p 0.8, top-k 20,
/// presence penalty 1.5. Neither mode uses `min_p` or a repetition penalty.
pub fn samplingDefaults(effort: Effort) sampling.Options {
    return switch (effort) {
        .off => .{ .temperature = 0.7, .top_p = 0.8, .top_k = 20, .min_p = 0, .presence_penalty = 1.5, .repetition_penalty = 1 },
        .low, .medium, .high, .xhigh => .{ .temperature = 1.0, .top_p = 0.95, .top_k = 20, .min_p = 0, .presence_penalty = 0, .repetition_penalty = 1 },
    };
}

/// The mode's defaults with the caller's overrides applied per option.
pub fn samplingOptions(effort: Effort, overrides: SamplingOverrides) sampling.Options {
    return samplingDefaults(effort).override(overrides);
}

fn trim(text: []const u8) []const u8 {
    return std.mem.trim(u8, text, " \t\r\n\x0b\x0c");
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

test "the prefix is a byte prefix of every rendering that starts with its system messages" {
    const alloc = std.testing.allocator;
    const tool: profiles.ToolDefinition = .{ .name = "read_file", .description = "Read a file.", .parameters = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}" };
    const system: Message = .{ .role = .system, .content = "You are terse." };
    const user: Message = .{ .role = .user, .content = "hello" };
    inline for (.{ .off, .low, .xhigh }) |effort| {
        for ([_][]const profiles.ToolDefinition{ &.{}, &.{tool} }) |tools| {
            const head = try prefix(alloc, &.{system}, tools, effort, .{});
            defer alloc.free(head);
            const full = try render(alloc, &.{ system, user }, tools, effort, .{});
            defer alloc.free(full);
            try std.testing.expect(head.len > 0);
            try std.testing.expect(std.mem.startsWith(u8, full, head));
            // The prefix ends at a turn boundary, so the remainder encodes on
            // its own the way the whole would.
            try std.testing.expect(std.mem.startsWith(u8, full[head.len..], "<|im_start|>user\n"));
        }
    }
    // No system message and no tools: nothing to prime with.
    const empty = try prefix(alloc, &.{}, &.{}, .off, .{});
    defer alloc.free(empty);
    try std.testing.expectEqual(@as(usize, 0), empty.len);
}

test "text prompts match pinned reference fixtures in every reasoning mode" {
    const Fixture = struct {
        template_sha256: []const u8,
        preserve_thinking: bool,
        prompt_cases: []const struct { name: []const u8, effort: Effort, messages: []const Message, prompt: []const u8 },
    };
    const alloc = std.testing.allocator;
    const fixture = try std.json.parseFromSlice(Fixture, alloc, @embedFile("fixtures/qwen38-text.json"), .{ .ignore_unknown_fields = true });
    defer fixture.deinit();
    try std.testing.expectEqualStrings(template_sha256, fixture.value.template_sha256);
    try std.testing.expect(fixture.value.preserve_thinking);
    try std.testing.expectEqual(@as(usize, 35), fixture.value.prompt_cases.len);
    var high_cases: usize = 0;
    for (fixture.value.prompt_cases) |case| {
        const prompt = try render(alloc, case.messages, &.{}, case.effort, .{});
        defer alloc.free(prompt);
        try std.testing.expectEqualStrings(case.prompt, prompt);
        // The template folds `high` into `xhigh`: the captured line says so.
        if (case.effort == .high) {
            high_cases += 1;
            try std.testing.expect(std.mem.indexOf(u8, case.prompt, "Reasoning effort is set to xhigh.") != null);
        }
    }
    try std.testing.expectEqual(@as(usize, 7), high_cases);
}

test "native tool prompts match pinned reference fixtures in every reasoning mode" {
    const Fixture = struct {
        template_sha256: []const u8,
        cases: []const struct {
            name: []const u8,
            effort: Effort,
            messages: []const Message,
            tools: []const profiles.ToolDefinition,
            prompt: []const u8,
        },
    };
    const alloc = std.testing.allocator;
    const fixture = try std.json.parseFromSlice(Fixture, alloc, @embedFile("fixtures/qwen38-tools.json"), .{ .ignore_unknown_fields = true });
    defer fixture.deinit();
    try std.testing.expectEqualStrings(template_sha256, fixture.value.template_sha256);
    try std.testing.expectEqual(@as(usize, 20), fixture.value.cases.len);
    for (fixture.value.cases) |case| {
        const prompt = try render(alloc, case.messages, case.tools, case.effort, .{});
        defer alloc.free(prompt);
        try std.testing.expectEqualStrings(case.prompt, prompt);
    }
}

test "parseTool round-trips a rendered call and rejects malformed bodies" {
    const alloc = std.testing.allocator;
    const calls = [_]profiles.ToolCall{
        .{ .id = 1, .name = "read_file", .arguments = "{\"path\":\"a.zig\",\"offset\":3,\"flag\":true}" },
    };
    const messages = [_]Message{
        .{ .role = .user, .content = "read it" },
        .{ .role = .assistant, .content = "", .tool_calls = &calls },
        .{ .role = .tool, .content = "body", .tool_call_id = 1 },
        .{ .role = .user, .content = "again" },
    };
    const prompt = try render(alloc, &messages, &.{}, .off, .{});
    defer alloc.free(prompt);
    const open_marker = "<tool_call>\n";
    const open = std.mem.indexOf(u8, prompt, open_marker).? + open_marker.len;
    const close = std.mem.indexOf(u8, prompt, "</tool_call>").?;
    const parsed = (try parseTool(alloc, prompt[open..close])).?;
    defer {
        alloc.free(parsed.name);
        alloc.free(parsed.arguments);
    }
    try std.testing.expectEqualStrings("read_file", parsed.name);
    try std.testing.expectEqualStrings("{\"path\":\"a.zig\",\"offset\":3,\"flag\":true}", parsed.arguments);

    try std.testing.expect((try parseTool(alloc, "no markers here")) == null);
    try std.testing.expect((try parseTool(alloc, "<parameter=x>1</parameter>")) == null);
    try std.testing.expect((try parseTool(alloc, "<function=>\n")) == null);
}

test "parseTool strips only the delimiters: a value keeps its trailing newline and leading spaces, a padded number is still a number" {
    const alloc = std.testing.allocator;
    const body = "<function=write_file>\n<parameter=path>\nnote.txt\n</parameter>\n<parameter=content>\n  indented\nlast line\n\n</parameter>\n<parameter=count>\n 20 \n</parameter>\n</function>";
    const parsed = (try parseTool(alloc, body)).?;
    defer {
        alloc.free(parsed.name);
        alloc.free(parsed.arguments);
    }
    try std.testing.expectEqualStrings("{\"path\":\"note.txt\",\"content\":\"  indented\\nlast line\\n\",\"count\":20}", parsed.arguments);

    // A rendered call round-trips its content byte for byte, newline included.
    const calls = [_]profiles.ToolCall{.{ .id = 1, .name = "write_file", .arguments = "{\"path\":\"n.txt\",\"content\":\"a\\n\"}" }};
    const messages = [_]Message{
        .{ .role = .user, .content = "write" },
        .{ .role = .assistant, .content = "", .tool_calls = &calls },
        .{ .role = .tool, .content = "ok", .tool_call_id = 1 },
        .{ .role = .user, .content = "again" },
    };
    const prompt = try render(alloc, &messages, &.{}, .off, .{});
    defer alloc.free(prompt);
    const open_marker = "<tool_call>\n";
    const open = std.mem.indexOf(u8, prompt, open_marker).? + open_marker.len;
    const close = std.mem.indexOf(u8, prompt, "</tool_call>").?;
    const back = (try parseTool(alloc, prompt[open..close])).?;
    defer {
        alloc.free(back.name);
        alloc.free(back.arguments);
    }
    try std.testing.expectEqualStrings(calls[0].arguments, back.arguments);
}

test "invalid conversations and bounded rendering return typed errors" {
    const alloc = std.testing.allocator;
    const user: Message = .{ .role = .user, .content = "Hi" };
    try std.testing.expectError(error.InvalidConversation, render(alloc, &.{}, &.{}, .off, .{}));
    try std.testing.expectError(error.InvalidConversation, render(alloc, &.{.{ .role = .assistant, .content = "x" }}, &.{}, .off, .{}));
    try std.testing.expectError(error.InvalidConversation, render(alloc, &.{ user, .{ .role = .system, .content = "x" }, user }, &.{}, .off, .{}));
    try std.testing.expectError(error.InvalidUtf8, render(alloc, &.{.{ .role = .user, .content = "\xff" }}, &.{}, .off, .{}));
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{.{ .role = .user, .content = "x", .reasoning_content = "y" }}, &.{}, .off, .{}));
    const stripped = try render(alloc, &.{ user, .{ .role = .assistant, .content = "<think>x</think>y<tool_call>z" }, user }, &.{}, .off, .{});
    defer alloc.free(stripped);
    try std.testing.expect(std.mem.indexOf(u8, stripped, "<|im_start|>assistant\n<think>\n\n</think>\n\nxyz<|im_end|>") != null);
    try std.testing.expectError(error.LimitExceeded, render(alloc, &.{user}, &.{}, .off, .{ .messages = 0 }));
    try std.testing.expectError(error.LimitExceeded, render(alloc, &.{user}, &.{}, .off, .{ .input_bytes = 1 }));
    try std.testing.expectError(error.LimitExceeded, render(alloc, &.{user}, &.{}, .off, .{ .output_bytes = 8 }));
    const prompt = try render(alloc, &.{user}, &.{}, .off, .{});
    defer alloc.free(prompt);
    const exact = try render(alloc, &.{user}, &.{}, .off, .{ .input_bytes = 2, .output_bytes = prompt.len });
    defer alloc.free(exact);
    try std.testing.expectEqualStrings(prompt, exact);
    try std.testing.expectError(error.LimitExceeded, render(alloc, &.{user}, &.{}, .off, .{ .output_bytes = prompt.len - 1 }));
}

test "sampling defaults follow the official per-mode table and flags override per option" {
    const instruct = samplingDefaults(.off);
    try std.testing.expectEqual(sampling.Options{ .temperature = 0.7, .top_p = 0.8, .top_k = 20, .min_p = 0, .presence_penalty = 1.5, .repetition_penalty = 1 }, instruct);
    const thinking = sampling.Options{ .temperature = 1.0, .top_p = 0.95, .top_k = 20, .min_p = 0, .presence_penalty = 0, .repetition_penalty = 1 };
    for ([_]Effort{ .low, .medium, .high, .xhigh }) |effort| try std.testing.expectEqual(thinking, samplingDefaults(effort));
    // Every profile validates as sampler options.
    for ([_]Effort{ .off, .low, .medium, .high, .xhigh }) |effort| _ = try sampling.Sampler.init(0, samplingDefaults(effort));
    // No overrides: the profile verbatim. One override: only that option moves.
    try std.testing.expectEqual(instruct, samplingOptions(.off, .{}));
    const greedy = samplingOptions(.off, .{ .temperature = 0 });
    try std.testing.expectEqual(@as(f32, 0), greedy.temperature);
    try std.testing.expectEqual(@as(f32, 1.5), greedy.presence_penalty);
    const custom = samplingOptions(.xhigh, .{ .top_k = 40, .min_p = 0.05, .presence_penalty = 0.5, .repetition_penalty = 1.1, .top_p = 0.9 });
    try std.testing.expectEqual(sampling.Options{ .temperature = 1.0, .top_p = 0.9, .top_k = 40, .min_p = 0.05, .presence_penalty = 0.5, .repetition_penalty = 1.1 }, custom);
}

fn allocationCase(alloc: std.mem.Allocator) !void {
    const prompt = try render(alloc, &.{ .{ .role = .system, .content = "Be precise." }, .{ .role = .user, .content = "Explain slices." } }, &.{}, .xhigh, .{});
    defer alloc.free(prompt);
}

test "prompt rendering cleans up every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationCase, .{});
}
