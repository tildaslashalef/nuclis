//! Bounded prompt profile for the pinned Gemma 4 artifacts, implemented from
//! the chat template embedded in the GGUF and verified against prompts the
//! pinned llama.cpp server rendered from it. The template text is not in the
//! tree; `fixtures/gemma4-text.json` is its evidence. The behavior it encodes
//! — turns, the thinking switch, dropped reasoning, and the `<bos>` prefix —
//! is specified with fixture-backed clauses in docs/reference/prompt-profile.md.
//!
//! Tool calling is the template's own DSL (`fixtures/gemma4-tools.json`,
//! docs/reference/tool-calling.md): declarations in the system turn, and
//! calls, results, and the follow-up all inside one `model` turn. The model
//! hands off by emitting `<|tool_response>`, which is therefore a stop token.
//! It owns no tokenizer or model equations.
const std = @import("std");
const sampling = @import("../sampling/root.zig");

/// Exact GGUF template this profile implements (`tokenizer.chat_template`,
/// 18,922 bytes ending in a newline); `profiles.forDocument` selects the
/// profile by this digest, never by architecture.
pub const template_sha256 = "845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b";
/// Other revisions of the same template that render every fixture case
/// byte for byte like the pinned one, so a file carrying one of them takes
/// this profile; `scripts/profile-alias-check.py` is the gate. None so far:
/// the 17,530-byte revision Gemma 4 finetunes ship differs on history
/// (prompt-profile.md § Evidence), so such a file needs `--prompt-profile`.
pub const template_aliases = [_][]const u8{};

const profiles = @import("root.zig");
pub const Role = profiles.Role;
pub const Effort = profiles.Effort;
pub const Message = profiles.Message;
pub const Limits = profiles.Limits;
pub const Error = profiles.Error;
pub const SamplingOverrides = profiles.SamplingOverrides;

/// The tokens that end a generation: `<turn|>` (106, the K-quant file's
/// `eos_token_id`), `<eos>` (1, the QAT file's), and `<|tool_response>` (50),
/// which the model emits after its calls to hand execution to the host.
pub const stop_tokens = [_][]const u8{ "<turn|>", "<eos>", "<|tool_response>" };

/// How the model delimits its reasoning in generated text: a thought
/// channel opened by `<|channel>thought\n` and closed by `<channel|>`. Both
/// markers are user-defined tokens the decoder always renders as text.
pub const reasoning: profiles.Reasoning = .{ .open = "<|channel>thought\n", .close = "<channel|>" };
pub const stream_markers: profiles.StreamMarkers = .{
    .open = "<|channel>",
    .close = "<channel|>",
    .open_suffix = "thought\n",
    .tool_open = "<|tool_call>",
    .tool_close = "<tool_call|>",
    .tool_parse = parseTool,
};

/// The DSL's string delimiter: strings are literal between two of these and
/// have no escape, so a string containing one cannot be represented.
const quote = "<|\"|>";
/// Every control marker the template spells inside a turn. Content, reasoning,
/// names, arguments, and results carrying one are rejected rather than
/// rendered, so structure cannot be smuggled past the profile.
const markers = [_][]const u8{ "<|channel>", "<channel|>", "<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>", quote };
/// Nesting the call parser accepts before it refuses the body.
const max_depth = 16;

/// Returns an owned UTF-8 prompt; caller frees it with the supplied allocator.
/// All message strings are borrowed only for this call. `profiles.validate`
/// enforces the shared conversation rules (completion-ready ending, role
/// order, bounds, tool history). Reasoning is trimmed before rendering, the
/// one departure from the template (which does not trim): the model writes
/// one newline before `<channel|>`, so trimmed text plus the template's
/// `\n<channel|>` reproduces the model's own bytes and the session's
/// incremental prefill hits. Content or reasoning carrying a control marker is
/// rejected rather than stripped as the template would.
/// The bytes every rendering that starts with these system messages and tools
/// starts with: `<bos>` and the system turn. Pinned as a byte prefix of `render`.
pub fn prefix(alloc: std.mem.Allocator, messages: []const Message, tools: []const profiles.ToolDefinition, effort: Effort, limits: Limits) Error![]u8 {
    var builder: Builder = .{ .alloc = alloc, .limit = limits.output_bytes };
    errdefer builder.bytes.deinit(alloc);
    try renderSystem(&builder, alloc, messages, tools, effort != .off);
    return builder.bytes.toOwnedSlice(alloc);
}

pub fn render(alloc: std.mem.Allocator, messages: []const Message, tools: []const profiles.ToolDefinition, effort: Effort, limits: Limits) Error![]u8 {
    try profiles.validate(alloc, messages, tools, limits);
    for (messages) |message| switch (message.role) {
        .assistant => {
            if (hasMarker(message.content) or hasMarker(message.reasoning_content)) return error.UnsupportedContent;
            for (message.tool_calls) |call| try checkName(call.name);
        },
        .tool => if (hasMarker(message.content)) return error.UnsupportedContent,
        else => {},
    };
    for (tools) |tool| try checkName(tool.name);
    const prefix_count = profiles.leadingSystemCount(messages);

    const thinking = effort != .off;
    var builder: Builder = .{ .alloc = alloc, .limit = limits.output_bytes };
    errdefer builder.bytes.deinit(alloc);
    try renderSystem(&builder, alloc, messages, tools, thinking);
    const rest = if (prefix_count > 0) messages[1..] else messages;

    // The template's reasoning gate: a thought is kept after the last user
    // message (the current loop) and on any message that calls tools (the
    // reference preserves reasoning by default for this template).
    var last_user: ?usize = null;
    for (rest, 0..) |message, i| {
        if (message.role == .user) last_user = i;
    }
    var previous: ?Role = null;
    // Whether the last rendered message ended on tool results, which leaves
    // the model turn open for the generation prompt.
    var open = false;
    for (rest, 0..) |message, i| {
        // Results are rendered by the assistant message they answer.
        if (message.role == .tool) continue;
        const continued = message.role == .assistant and previous == .assistant;
        if (!continued) {
            try builder.add("<|turn>");
            try builder.add(roleName(message.role));
            try builder.add("\n");
        }
        var responded = false;
        if (message.role == .assistant) {
            const reasoning_text = trim(message.reasoning_content);
            const gate = if (last_user) |u| i > u else true;
            if (reasoning_text.len != 0 and (gate or message.tool_calls.len != 0)) {
                try builder.add(reasoning.open);
                try builder.add(reasoning_text);
                try builder.add("\n");
                try builder.add(reasoning.close);
            }
            for (message.tool_calls) |call| try renderCall(&builder, alloc, call);
            // `validate` proved the results answer these calls in order.
            var k = i + 1;
            while (k < rest.len and rest[k].role == .tool) : (k += 1) {
                try renderResponse(&builder, message.tool_calls[k - i - 1].name, rest[k].content);
                responded = true;
            }
        }
        const content = trim(message.content);
        try builder.add(content);
        var next: ?Role = null;
        for (rest[i + 1 ..]) |later| {
            if (later.role != .tool) {
                next = later.role;
                break;
            }
        }
        const continues = message.role == .assistant and next == .assistant and (message.tool_calls.len == 0 or responded);
        if (!continues and !(responded and content.len == 0 and next == null)) try builder.add("<turn|>\n");
        previous = message.role;
        open = responded;
    }
    if (!open) {
        try builder.add("<|turn>model\n");
        if (!thinking) {
            try builder.add(reasoning.open);
            try builder.add(reasoning.close);
        }
    } else if (thinking) {
        try builder.add(reasoning.open);
    }
    // The reference's own repair: results followed by content close the turn
    // and the template adds no model turn, so it reopens one.
    if (std.mem.endsWith(u8, builder.bytes.items, "<turn|>\n")) try builder.add("<|turn>model\n");
    return builder.bytes.toOwnedSlice(alloc);
}

/// The leading `<bos>` and system turn, which exists for thinking, a leading
/// system message, or tool declarations. Only the first message is folded
/// into it; declarations follow the text with no separator.
fn renderSystem(builder: *Builder, alloc: std.mem.Allocator, messages: []const Message, tools: []const profiles.ToolDefinition, thinking: bool) Error!void {
    const prefix_count = profiles.leadingSystemCount(messages);
    try builder.add("<bos>");
    if (!thinking and prefix_count == 0 and tools.len == 0) return;
    try builder.add("<|turn>system\n");
    if (thinking) try builder.add("<|think|>\n");
    if (prefix_count > 0) try builder.add(trim(messages[0].content));
    for (tools) |tool| {
        try builder.add("<|tool>");
        try renderDeclaration(builder, alloc, tool);
        try builder.add("<tool|>");
    }
    try builder.add("<turn|>\n");
}

/// `validate` already accepted this JSON, so a parse failure here is an
/// allocation failure and must not be misreported.
fn jsonError(err: anyerror) Error {
    return if (err == error.OutOfMemory) error.OutOfMemory else error.InvalidConversation;
}

fn hasMarker(text: []const u8) bool {
    for (markers) |marker| {
        if (std.mem.indexOf(u8, text, marker) != null) return true;
    }
    return false;
}

/// A tool name is spelled bare before `{` in declarations, calls, and
/// results, so it can carry neither braces, whitespace, nor a marker.
fn checkName(name: []const u8) Error!void {
    if (hasMarker(name)) return error.UnsupportedContent;
    for (name) |byte| switch (byte) {
        '{', '}', ' ', '\t', '\r', '\n' => return error.UnsupportedContent,
        else => {},
    };
}

fn addQuoted(builder: *Builder, text: []const u8) Error!void {
    if (hasMarker(text)) return error.UnsupportedContent;
    try builder.add(quote);
    try builder.add(text);
    try builder.add(quote);
}

/// A JSON string's ASCII-uppercased form, as the template's `upper` filter
/// renders schema types; non-ASCII is left as it is.
fn addUpper(builder: *Builder, text: []const u8) Error!void {
    for (text) |byte| try builder.add(&.{std.ascii.toUpper(byte)});
}

/// One `declaration:NAME{…}` as the template's macro renders it: the
/// description, then `parameters` when the schema object is nonempty
/// (`properties`, `required`, and the mandatory `type`). The subset of JSON
/// Schema the macro understands is rendered; a schema outside it is rejected
/// rather than approximated.
fn renderDeclaration(builder: *Builder, alloc: std.mem.Allocator, tool: profiles.ToolDefinition) Error!void {
    try builder.add("declaration:");
    try builder.add(tool.name);
    try builder.add("{description:");
    try addQuoted(builder, tool.description);
    // `validate` already proved this is a JSON object.
    const parsed = std.json.parseFromSlice(std.json.Value, alloc, tool.parameters, .{ .parse_numbers = false }) catch |err| return jsonError(err);
    defer parsed.deinit();
    const params = parsed.value.object;
    if (params.count() != 0) {
        try builder.add(",parameters:{");
        if (nonEmptyObject(params.get("properties"))) |props| {
            try builder.add("properties:{");
            try renderProperties(builder, alloc, props, false, 0);
            try builder.add("},");
        }
        if (nonEmptyArray(params.get("required"))) |required| {
            try builder.add("required:[");
            try renderStringList(builder, required);
            try builder.add("],");
        }
        // Without a type the macro leaves its brace unclosed.
        const type_text = nonEmptyString(params.get("type")) orelse return error.UnsupportedContent;
        try builder.add("type:");
        try builder.add(quote);
        try addUpper(builder, type_text);
        try builder.add(quote);
        try builder.add("}");
    }
    try builder.add("}");
}

/// The macro's `format_parameters`: every property in byte order of its key
/// (the reference sorts case-sensitively), rendered as
/// `KEY:{description?,enum?|items?,nullable?,properties?,required?,type}`.
/// With `filter_keys` the standard schema keys are skipped: the template's
/// fallback for an object property without `properties`, which treats the
/// property's other keys as sub-properties.
fn renderProperties(builder: *Builder, alloc: std.mem.Allocator, props: std.json.ObjectMap, filter_keys: bool, depth: usize) Error!void {
    if (depth > max_depth) return error.UnsupportedContent;
    const keys = try sortedKeys(alloc, props);
    defer alloc.free(keys);
    var first = true;
    for (keys) |key| {
        if (filter_keys and isStandardKey(key)) continue;
        const value = props.get(key).?;
        if (value != .object) return error.UnsupportedContent;
        const fields = value.object;
        if (!first) try builder.add(",");
        first = false;
        if (hasMarker(key)) return error.UnsupportedContent;
        try builder.add(key);
        try builder.add(":{");
        var comma = false;
        if (nonEmptyString(fields.get("description"))) |description| {
            try builder.add("description:");
            try addQuoted(builder, description);
            comma = true;
        }
        const type_text = nonEmptyString(fields.get("type")) orelse return error.UnsupportedContent;
        var upper_buf: [32]u8 = undefined;
        if (type_text.len > upper_buf.len) return error.UnsupportedContent;
        const type_upper = std.ascii.upperString(&upper_buf, type_text);
        if (std.mem.eql(u8, type_upper, "STRING")) {
            if (nonEmptyArray(fields.get("enum"))) |values| {
                try separator(builder, &comma);
                try builder.add("enum:");
                try renderArgument(builder, alloc, .{ .array = values }, true, depth + 1);
            }
        } else if (std.mem.eql(u8, type_upper, "ARRAY")) {
            if (nonEmptyObject(fields.get("items"))) |items| {
                try separator(builder, &comma);
                try builder.add("items:{");
                try renderItems(builder, alloc, items, depth + 1);
                try builder.add("}");
            }
        }
        if (fields.get("nullable")) |nullable| {
            if (nullable != .bool) return error.UnsupportedContent;
            if (nullable.bool) {
                try separator(builder, &comma);
                try builder.add("nullable:true");
            }
        }
        if (std.mem.eql(u8, type_upper, "OBJECT")) {
            try separator(builder, &comma);
            try builder.add("properties:{");
            if (fields.get("properties")) |nested| {
                if (nested != .object) return error.UnsupportedContent;
                try renderProperties(builder, alloc, nested.object, false, depth + 1);
            } else try renderProperties(builder, alloc, fields, true, depth + 1);
            try builder.add("}");
            if (nonEmptyArray(fields.get("required"))) |required| {
                try separator(builder, &comma);
                try builder.add("required:[");
                try renderStringList(builder, required);
                try builder.add("]");
            }
        }
        try separator(builder, &comma);
        try builder.add("type:");
        try addQuoted(builder, type_upper);
        try builder.add("}");
    }
}

fn isStandardKey(key: []const u8) bool {
    inline for (.{ "description", "type", "properties", "required", "nullable" }) |standard| {
        if (std.mem.eql(u8, key, standard)) return true;
    }
    return false;
}

/// The macro's comma rule: nothing before the first field, a comma before
/// every later one.
fn separator(builder: *Builder, comma: *bool) Error!void {
    if (comma.*) try builder.add(",") else comma.* = true;
}

/// An array property's `items` schema: its keys in byte order, null values
/// skipped; `properties` and `required` recurse as schema, `type` is
/// uppercased, and any other key renders as a plain argument.
fn renderItems(builder: *Builder, alloc: std.mem.Allocator, items: std.json.ObjectMap, depth: usize) Error!void {
    if (depth > max_depth) return error.UnsupportedContent;
    const keys = try sortedKeys(alloc, items);
    defer alloc.free(keys);
    var first = true;
    for (keys) |key| {
        const value = items.get(key).?;
        if (value == .null) continue;
        if (!first) try builder.add(",");
        first = false;
        if (std.mem.eql(u8, key, "properties")) {
            try builder.add("properties:{");
            if (value == .object) try renderProperties(builder, alloc, value.object, false, depth + 1);
            try builder.add("}");
        } else if (std.mem.eql(u8, key, "required")) {
            if (value != .array) return error.UnsupportedContent;
            try builder.add("required:[");
            try renderStringList(builder, value.array);
            try builder.add("]");
        } else if (std.mem.eql(u8, key, "type")) {
            try builder.add("type:");
            switch (value) {
                .string => |text| {
                    try builder.add(quote);
                    try addUpper(builder, text);
                    try builder.add(quote);
                },
                .array => |types| {
                    try builder.add("[");
                    for (types.items, 0..) |item, i| {
                        if (item != .string) return error.UnsupportedContent;
                        if (i != 0) try builder.add(",");
                        try builder.add(quote);
                        try addUpper(builder, item.string);
                        try builder.add(quote);
                    }
                    try builder.add("]");
                },
                else => return error.UnsupportedContent,
            }
        } else {
            if (hasMarker(key)) return error.UnsupportedContent;
            try builder.add(key);
            try builder.add(":");
            try renderArgument(builder, alloc, value, true, depth + 1);
        }
    }
}

/// `required:[…]` items: quoted strings joined by commas.
fn renderStringList(builder: *Builder, list: std.json.Array) Error!void {
    for (list.items, 0..) |item, i| {
        if (item != .string) return error.UnsupportedContent;
        if (i != 0) try builder.add(",");
        try addQuoted(builder, item.string);
    }
}

/// The macro's `format_argument`: strings quoted, numbers as their JSON
/// text, `true`/`false`/`null`, objects `{k:v,…}` in byte order of their keys
/// (quoted with `escape_keys`, bare otherwise), arrays `[v,…]`.
fn renderArgument(builder: *Builder, alloc: std.mem.Allocator, value: std.json.Value, escape_keys: bool, depth: usize) Error!void {
    if (depth > max_depth) return error.UnsupportedContent;
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
        .string => |text| try addQuoted(builder, text),
        .array => |array| {
            try builder.add("[");
            for (array.items, 0..) |item, i| {
                if (i != 0) try builder.add(",");
                try renderArgument(builder, alloc, item, escape_keys, depth + 1);
            }
            try builder.add("]");
        },
        .object => |object| {
            try builder.add("{");
            const keys = try sortedKeys(alloc, object);
            defer alloc.free(keys);
            for (keys, 0..) |key, i| {
                if (i != 0) try builder.add(",");
                if (escape_keys) try addQuoted(builder, key) else {
                    if (hasMarker(key)) return error.UnsupportedContent;
                    try builder.add(key);
                }
                try builder.add(":");
                try renderArgument(builder, alloc, object.get(key).?, escape_keys, depth + 1);
            }
            try builder.add("}");
        },
    }
}

/// The object's keys in byte order: the reference's `dictsort` is
/// case-sensitive. Caller frees the slice; the keys borrow the map.
fn sortedKeys(alloc: std.mem.Allocator, object: std.json.ObjectMap) std.mem.Allocator.Error![][]const u8 {
    const keys = try alloc.dupe([]const u8, object.keys());
    std.mem.sort([]const u8, keys, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lessThan);
    return keys;
}

// Jinja truthiness for the schema fields the macro tests.
fn nonEmptyObject(value: ?std.json.Value) ?std.json.ObjectMap {
    const v = value orelse return null;
    if (v != .object or v.object.count() == 0) return null;
    return v.object;
}
fn nonEmptyArray(value: ?std.json.Value) ?std.json.Array {
    const v = value orelse return null;
    if (v != .array or v.array.items.len == 0) return null;
    return v.array;
}
fn nonEmptyString(value: ?std.json.Value) ?[]const u8 {
    const v = value orelse return null;
    if (v != .string or v.string.len == 0) return null;
    return v.string;
}

/// One assistant call in the template's syntax: `<|tool_call>call:NAME{k:v,…}<tool_call|>`
/// with keys in byte order and values as `format_argument` renders them.
fn renderCall(builder: *Builder, alloc: std.mem.Allocator, call: profiles.ToolCall) Error!void {
    try builder.add("<|tool_call>call:");
    try builder.add(call.name);
    try builder.add("{");
    // `validate` already proved these arguments are a JSON object.
    const parsed = std.json.parseFromSlice(std.json.Value, alloc, call.arguments, .{ .parse_numbers = false }) catch |err| return jsonError(err);
    defer parsed.deinit();
    const keys = try sortedKeys(alloc, parsed.value.object);
    defer alloc.free(keys);
    for (keys, 0..) |key, i| {
        if (i != 0) try builder.add(",");
        if (hasMarker(key)) return error.UnsupportedContent;
        try builder.add(key);
        try builder.add(":");
        try renderArgument(builder, alloc, parsed.value.object.get(key).?, false, 0);
    }
    try builder.add("}<tool_call|>");
}

/// One result, inside the model turn that called for it. The content is a
/// string, so it is the template's `{value:<|"|>…<|"|>}`, untrimmed.
fn renderResponse(builder: *Builder, name: []const u8, content: []const u8) Error!void {
    try builder.add("<|tool_response>response:");
    try builder.add(name);
    try builder.add("{value:");
    try addQuoted(builder, content);
    try builder.add("}<tool_response|>");
}

/// Parses one native call body — the text between the `<|tool_call>` and
/// `<tool_call|>` control tokens, `call:NAME{…}` — into an owned `ToolCall`,
/// or null when the body is not one complete, well-formed call. The inverse
/// of `renderCall`: strings are literal between delimiters, numbers keep
/// their text, nested objects and arrays keep their wire order. Whitespace is
/// allowed where the reference grammar allows it (after `{`, `[`, `,`, `:`).
pub fn parseTool(alloc: std.mem.Allocator, body: []const u8) std.mem.Allocator.Error!?profiles.ToolCall {
    const text = std.mem.trim(u8, body, " \t\r\n");
    const call_marker = "call:";
    if (!std.mem.startsWith(u8, text, call_marker)) return null;
    const brace = std.mem.indexOfScalarPos(u8, text, call_marker.len, '{') orelse return null;
    const name = text[call_marker.len..brace];
    if (name.len == 0) return null;
    checkName(name) catch return null;

    var arguments: std.Io.Writer.Allocating = .init(alloc);
    // Also frees the partial JSON on every null return; the success path
    // takes ownership of the bytes first.
    defer arguments.deinit();
    var parser: Parser = .{ .text = text, .pos = brace, .json = .{ .writer = &arguments.writer } };
    if (!try parser.object(0)) return null;
    parser.space();
    if (parser.pos != text.len) return null;
    const name_owned = try alloc.dupe(u8, name);
    errdefer alloc.free(name_owned);
    const arguments_owned = try arguments.toOwnedSlice();
    errdefer alloc.free(arguments_owned);
    return .{ .id = 0, .name = name_owned, .arguments = arguments_owned };
}

/// A recursive-descent reader of the call DSL that writes JSON as it goes.
/// Every method returns false for a malformed or truncated body; the only
/// error is allocation (the writer is allocating, so its failure is OOM).
const Parser = struct {
    text: []const u8,
    pos: usize,
    json: std.json.Stringify,

    const Fail = std.mem.Allocator.Error;

    fn space(self: *Parser) void {
        while (self.pos < self.text.len and std.ascii.isWhitespace(self.text[self.pos])) self.pos += 1;
    }

    fn take(self: *Parser, literal: []const u8) bool {
        if (!std.mem.startsWith(u8, self.text[self.pos..], literal)) return false;
        self.pos += literal.len;
        return true;
    }

    fn object(self: *Parser, depth: usize) Fail!bool {
        if (depth > max_depth or !self.take("{")) return false;
        self.json.beginObject() catch return error.OutOfMemory;
        self.space();
        if (self.take("}")) {
            self.json.endObject() catch return error.OutOfMemory;
            return true;
        }
        while (true) {
            // A key is anything up to the colon, bar a closing brace.
            const start = self.pos;
            while (self.pos < self.text.len and self.text[self.pos] != ':' and self.text[self.pos] != '}') self.pos += 1;
            if (self.pos == start or self.pos == self.text.len or self.text[self.pos] != ':') return false;
            self.json.objectField(self.text[start..self.pos]) catch return error.OutOfMemory;
            self.pos += 1;
            self.space();
            if (!try self.value(depth)) return false;
            self.space();
            if (self.take("}")) break;
            if (!self.take(",")) return false;
            self.space();
        }
        self.json.endObject() catch return error.OutOfMemory;
        return true;
    }

    fn array(self: *Parser, depth: usize) Fail!bool {
        if (depth > max_depth or !self.take("[")) return false;
        self.json.beginArray() catch return error.OutOfMemory;
        self.space();
        if (self.take("]")) {
            self.json.endArray() catch return error.OutOfMemory;
            return true;
        }
        while (true) {
            if (!try self.value(depth)) return false;
            self.space();
            if (self.take("]")) break;
            if (!self.take(",")) return false;
            self.space();
        }
        self.json.endArray() catch return error.OutOfMemory;
        return true;
    }

    fn value(self: *Parser, depth: usize) Fail!bool {
        if (self.pos >= self.text.len) return false;
        if (self.take(quote)) {
            const end = std.mem.indexOfPos(u8, self.text, self.pos, quote) orelse return false;
            self.json.write(self.text[self.pos..end]) catch return error.OutOfMemory;
            self.pos = end + quote.len;
            return true;
        }
        switch (self.text[self.pos]) {
            '{' => return self.object(depth + 1),
            '[' => return self.array(depth + 1),
            't' => {
                if (!self.take("true")) return false;
                self.json.write(true) catch return error.OutOfMemory;
            },
            'f' => {
                if (!self.take("false")) return false;
                self.json.write(false) catch return error.OutOfMemory;
            },
            'n' => {
                if (!self.take("null")) return false;
                self.json.write(null) catch return error.OutOfMemory;
            },
            else => return self.number(),
        }
        return true;
    }

    /// A JSON number, copied as written.
    fn number(self: *Parser) Fail!bool {
        const start = self.pos;
        _ = self.take("-");
        const int_start = self.pos;
        while (self.pos < self.text.len and std.ascii.isDigit(self.text[self.pos])) self.pos += 1;
        const int_len = self.pos - int_start;
        if (int_len == 0 or (int_len > 1 and self.text[int_start] == '0')) return false;
        if (self.take(".")) {
            const frac = self.pos;
            while (self.pos < self.text.len and std.ascii.isDigit(self.text[self.pos])) self.pos += 1;
            if (self.pos == frac) return false;
        }
        if (self.pos < self.text.len and (self.text[self.pos] == 'e' or self.text[self.pos] == 'E')) {
            self.pos += 1;
            if (!self.take("+")) _ = self.take("-");
            const exp = self.pos;
            while (self.pos < self.text.len and std.ascii.isDigit(self.text[self.pos])) self.pos += 1;
            if (self.pos == exp) return false;
        }
        self.json.print("{s}", .{self.text[start..self.pos]}) catch return error.OutOfMemory;
        return true;
    }
};

/// The template's role names: the assistant is `model`, and the reference
/// renders `developer` as `system`.
fn roleName(role: Role) []const u8 {
    return switch (role) {
        .system, .developer => "system",
        .user => "user",
        .assistant => "model",
        // Unreachable: results are rendered inside the model turn, never named.
        .tool => unreachable,
    };
}

/// The checkpoint's own sampling hint, read from the K-quant file's header
/// on 2026-09-11 (`general.sampling.temp` 1.0, `general.sampling.top_p`
/// 0.95, `general.sampling.top_k` 64; the QAT file carries the same keys).
/// The header states one setting for both thinking modes; no penalties and
/// no `min_p` are declared. Recorded as the file's claim: no published
/// per-mode table was pinned for it, unlike Qwen3.8's.
pub fn samplingDefaults(effort: Effort) sampling.Options {
    _ = effort;
    return .{ .temperature = 1.0, .top_p = 0.95, .top_k = 64, .min_p = 0, .presence_penalty = 0, .repetition_penalty = 1 };
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

const Fixture = struct {
    template_sha256: []const u8,
    preserve_thinking: bool,
    prompt_cases: []const struct { name: []const u8, effort: Effort, messages: []const Message, prompt: []const u8, tokens: []const u32 },
};

test "text prompts match the pinned reference fixtures, `<bos>` prepended" {
    const alloc = std.testing.allocator;
    const fixture = try std.json.parseFromSlice(Fixture, alloc, @embedFile("fixtures/gemma4-text.json"), .{ .ignore_unknown_fields = true });
    defer fixture.deinit();
    try std.testing.expectEqualStrings(template_sha256, fixture.value.template_sha256);
    try std.testing.expect(!fixture.value.preserve_thinking);
    try std.testing.expectEqual(@as(usize, 14), fixture.value.prompt_cases.len);
    for (fixture.value.prompt_cases) |case| {
        const expected = try std.mem.concat(alloc, u8, &.{ "<bos>", case.prompt });
        defer alloc.free(expected);
        const prompt = try render(alloc, case.messages, &.{}, case.effort, .{});
        defer alloc.free(prompt);
        try std.testing.expectEqualStrings(expected, prompt);
        // The captured token stream starts with `<|turn>` (105) and every
        // fixture prompt is the reference's without its BOS.
        try std.testing.expectEqual(@as(u32, 105), case.tokens[0]);
    }
}

test "thinking is a switch: every effort but off renders like medium" {
    const alloc = std.testing.allocator;
    const messages = [_]Message{ .{ .role = .system, .content = "Be brief." }, .{ .role = .user, .content = "Hi" } };
    const on = try render(alloc, &messages, &.{}, .medium, .{});
    defer alloc.free(on);
    for ([_]Effort{ .low, .xhigh }) |effort| {
        const other = try render(alloc, &messages, &.{}, effort, .{});
        defer alloc.free(other);
        try std.testing.expectEqualStrings(on, other);
    }
    const off = try render(alloc, &messages, &.{}, .off, .{});
    defer alloc.free(off);
    try std.testing.expectEqualStrings("<bos><|turn>system\n<|think|>\nBe brief.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n", on);
    try std.testing.expectEqualStrings("<bos><|turn>system\nBe brief.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>", off);
}

test "history drops assistant reasoning, as the template's gate does" {
    const alloc = std.testing.allocator;
    const with = try render(alloc, &.{ .{ .role = .user, .content = "a" }, .{ .role = .assistant, .content = "b", .reasoning_content = "hidden" }, .{ .role = .user, .content = "c" } }, &.{}, .medium, .{});
    defer alloc.free(with);
    const without = try render(alloc, &.{ .{ .role = .user, .content = "a" }, .{ .role = .assistant, .content = "b" }, .{ .role = .user, .content = "c" } }, &.{}, .medium, .{});
    defer alloc.free(without);
    try std.testing.expectEqualStrings(without, with);
    try std.testing.expect(std.mem.indexOf(u8, with, "hidden") == null);
}

test "invalid conversations and bounded rendering return typed errors" {
    const alloc = std.testing.allocator;
    const user: Message = .{ .role = .user, .content = "Hi" };
    try std.testing.expectError(error.InvalidConversation, render(alloc, &.{}, &.{}, .off, .{}));
    try std.testing.expectError(error.InvalidConversation, render(alloc, &.{.{ .role = .assistant, .content = "x" }}, &.{}, .off, .{}));
    try std.testing.expectError(error.InvalidConversation, render(alloc, &.{ user, .{ .role = .system, .content = "x" }, user }, &.{}, .off, .{}));
    try std.testing.expectError(error.InvalidUtf8, render(alloc, &.{.{ .role = .user, .content = "\xff" }}, &.{}, .off, .{}));
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{.{ .role = .user, .content = "x", .reasoning_content = "y" }}, &.{}, .off, .{}));
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{ .{ .role = .assistant, .content = "<|channel>thought\nx<channel|>y" }, user }, &.{}, .off, .{}));
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{ .{ .role = .assistant, .content = "x<channel|>" }, user }, &.{}, .off, .{}));
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

test "sampling defaults are the file's hint in every mode and flags override per option" {
    const hint = sampling.Options{ .temperature = 1.0, .top_p = 0.95, .top_k = 64, .min_p = 0, .presence_penalty = 0, .repetition_penalty = 1 };
    for ([_]Effort{ .off, .low, .medium, .xhigh }) |effort| {
        try std.testing.expectEqual(hint, samplingDefaults(effort));
        _ = try sampling.Sampler.init(0, samplingDefaults(effort));
    }
    try std.testing.expectEqual(hint, samplingOptions(.off, .{}));
    const greedy = samplingOptions(.medium, .{ .temperature = 0 });
    try std.testing.expectEqual(@as(f32, 0), greedy.temperature);
    try std.testing.expectEqual(@as(usize, 64), greedy.top_k);
}

fn allocationCase(alloc: std.mem.Allocator) !void {
    const prompt = try render(alloc, &.{ .{ .role = .system, .content = "Be precise." }, .{ .role = .user, .content = "Explain slices." } }, &.{}, .xhigh, .{});
    defer alloc.free(prompt);
    const call: profiles.ToolCall = .{ .id = 1, .name = "read_file", .arguments = "{\"path\":\"a\",\"n\":[1,{\"b\":true}]}" };
    const with_tools = try render(alloc, &.{
        .{ .role = .user, .content = "go" },
        .{ .role = .assistant, .content = "", .reasoning_content = "Plan.", .tool_calls = &.{call} },
        .{ .role = .tool, .content = "body", .tool_call_id = 1 },
    }, &.{read_tool}, .medium, .{});
    defer alloc.free(with_tools);
    if (try parseTool(alloc, "call:f{a:<|\"|>x<|\"|>,b:[1,{c:null}]}")) |parsed| {
        alloc.free(parsed.name);
        alloc.free(parsed.arguments);
    } else return error.UnexpectedNull;
}

test "prompt rendering cleans up every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationCase, .{});
}

test "the prefix is a byte prefix of every rendering that starts with its system message and tools" {
    const alloc = std.testing.allocator;
    const system: Message = .{ .role = .system, .content = "You are terse." };
    const user: Message = .{ .role = .user, .content = "hello" };
    inline for (.{ .off, .low }) |effort| {
        for ([_][]const Message{ &.{ system, user }, &.{user} }) |messages| {
            for ([_][]const profiles.ToolDefinition{ &.{}, &.{read_tool} }) |tools| {
                const head = try prefix(alloc, messages[0 .. messages.len - 1], tools, effort, .{});
                defer alloc.free(head);
                const full = try render(alloc, messages, tools, effort, .{});
                defer alloc.free(full);
                try std.testing.expect(std.mem.startsWith(u8, full, head));
                try std.testing.expect(std.mem.startsWith(u8, full[head.len..], "<|turn>user\n"));
            }
        }
    }
}

const read_tool: profiles.ToolDefinition = .{
    .name = "read_file",
    .description = "Read a file.",
    .parameters = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"offset\":{\"type\":\"integer\"}},\"required\":[\"path\"]}",
};

test "native tool prompts match the pinned reference fixtures, `<bos>` prepended" {
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
    const fixture = try std.json.parseFromSlice(ToolFixture, alloc, @embedFile("fixtures/gemma4-tools.json"), .{ .ignore_unknown_fields = true });
    defer fixture.deinit();
    try std.testing.expectEqualStrings(template_sha256, fixture.value.template_sha256);
    try std.testing.expect(fixture.value.preserve_thinking);
    try std.testing.expectEqual(@as(usize, 24), fixture.value.cases.len);
    for (fixture.value.cases) |case| {
        const expected = try std.mem.concat(alloc, u8, &.{ "<bos>", case.prompt });
        defer alloc.free(expected);
        const prompt = try render(alloc, case.messages, case.tools, case.effort, .{});
        defer alloc.free(prompt);
        try std.testing.expectEqualStrings(expected, prompt);
    }
}

test "reasoning is trimmed on a call step, so padded and unpadded text render alike" {
    const alloc = std.testing.allocator;
    const call: profiles.ToolCall = .{ .id = 1, .name = "read_file", .arguments = "{\"path\":\"a\"}" };
    const padded = [_]Message{
        .{ .role = .user, .content = "go" },
        .{ .role = .assistant, .content = "", .reasoning_content = "Plan.\n", .tool_calls = &.{call} },
        .{ .role = .tool, .content = "body", .tool_call_id = 1 },
    };
    const bare = [_]Message{ padded[0], .{ .role = .assistant, .content = "", .reasoning_content = "Plan.", .tool_calls = &.{call} }, padded[2] };
    const a = try render(alloc, &padded, &.{}, .medium, .{});
    defer alloc.free(a);
    const b = try render(alloc, &bare, &.{}, .medium, .{});
    defer alloc.free(b);
    try std.testing.expectEqualStrings(b, a);
    try std.testing.expect(std.mem.indexOf(u8, a, "<|channel>thought\nPlan.\n<channel|><|tool_call>") != null);
}

test "parseTool round-trips a rendered call and keeps nested values" {
    const alloc = std.testing.allocator;
    const arguments = "{\"query\":\"a\\\"b\\nc\",\"limit\":3,\"ratio\":1.5,\"regex\":true,\"nothing\":null,\"paths\":[\"x\",\"y\"],\"options\":{\"case_sensitive\":false}}";
    const messages = [_]Message{
        .{ .role = .user, .content = "search" },
        .{ .role = .assistant, .content = "", .tool_calls = &.{.{ .id = 1, .name = "search", .arguments = arguments }} },
        .{ .role = .tool, .content = "none", .tool_call_id = 1 },
        .{ .role = .user, .content = "ok" },
    };
    const prompt = try render(alloc, &messages, &.{}, .off, .{});
    defer alloc.free(prompt);
    const open_marker = "<|tool_call>";
    const open = std.mem.indexOf(u8, prompt, open_marker).? + open_marker.len;
    const close = std.mem.indexOf(u8, prompt, "<tool_call|>").?;
    const parsed = (try parseTool(alloc, prompt[open..close])).?;
    defer {
        alloc.free(parsed.name);
        alloc.free(parsed.arguments);
    }
    try std.testing.expectEqualStrings("search", parsed.name);
    // Keys come back in wire order, which the renderer sorted.
    try std.testing.expectEqualStrings("{\"limit\":3,\"nothing\":null,\"options\":{\"case_sensitive\":false},\"paths\":[\"x\",\"y\"],\"query\":\"a\\\"b\\nc\",\"ratio\":1.5,\"regex\":true}", parsed.arguments);
}

test "parseTool accepts the grammar's whitespace and refuses malformed or truncated bodies" {
    const alloc = std.testing.allocator;
    const spaced = (try parseTool(alloc, "call:f{ a: 1 , b:[ 1 , -2.5e3 ] , c:{ } }")).?;
    defer {
        alloc.free(spaced.name);
        alloc.free(spaced.arguments);
    }
    try std.testing.expectEqualStrings("{\"a\":1,\"b\":[1,-2.5e3],\"c\":{}}", spaced.arguments);
    const empty = (try parseTool(alloc, "call:bash{}")).?;
    defer {
        alloc.free(empty.name);
        alloc.free(empty.arguments);
    }
    try std.testing.expectEqualStrings("{}", empty.arguments);
    for ([_][]const u8{
        "no call here",
        "call:{a:1}", // empty name
        "call:f x{a:1}", // whitespace in the name
        "call:f{a:1", // truncated
        "call:f{a:<|\"|>open}", // unterminated string
        "call:f{a:1}x", // trailing text
        "call:f{:1}", // empty key
        "call:f{a:01}", // leading zero
        "call:f{a:1.}", // dangling fraction
        "call:f{a:tru}", // partial literal
        "call:f{a:[1,]}", // dangling comma
        "call:f{a:{b:1},}", // dangling comma
        "call:f{a:[[[[[[[[[[[[[[[[[[1]]]]]]]]]]]]]]]]]]}", // beyond the depth bound
    }) |body| {
        try std.testing.expect((try parseTool(alloc, body)) == null);
    }
}

test "the stream decoder drives the Gemma grammar end to end" {
    const alloc = std.testing.allocator;
    const Sink = struct {
        thinking: std.Io.Writer.Allocating,
        answer: std.Io.Writer.Allocating,
        calls: std.ArrayList(profiles.ToolCall) = .empty,
        alloc: std.mem.Allocator,
        pub fn send(self: *@This(), e: @import("../events.zig").Event) !void {
            switch (e) {
                .thinking => |t| try self.thinking.writer.writeAll(t),
                .answer => |t| try self.answer.writer.writeAll(t),
                .tool_call => |c| try self.calls.append(self.alloc, .{ .id = 0, .name = try self.alloc.dupe(u8, c.name), .arguments = try self.alloc.dupe(u8, c.arguments) }),
                .stop => return error.UnexpectedEvent,
            }
        }
    };
    var sink: Sink = .{ .thinking = .init(alloc), .answer = .init(alloc), .alloc = alloc };
    defer {
        for (sink.calls.items) |c| {
            alloc.free(c.name);
            alloc.free(c.arguments);
        }
        sink.calls.deinit(alloc);
        sink.thinking.deinit();
        sink.answer.deinit();
    }
    const stream = profiles.stream;
    var d = stream.Decoder.init(alloc, .{ .open = 100, .close = 101, .open_suffix = "thought\n", .tool = .{ .open = 48, .close = 49, .parse = parseTool } }, true);
    defer d.deinit();
    try d.feed(100, "<|channel>", &sink);
    try d.feed(7, "thought\n", &sink);
    try d.feed(7, "plan", &sink);
    try d.feed(101, "<channel|>", &sink);
    try d.feed(48, "<|tool_call>", &sink);
    for ([_][]const u8{ "call:", "read_file", "{path:", "<|\"|>", "a.zig", "<|\"|>", ",offset:", "3", "}" }) |piece| try d.feed(7, piece, &sink);
    try d.feed(49, "<tool_call|>", &sink);
    try d.feed(48, "<|tool_call>", &sink);
    try d.feed(7, "call:bash{command:<|\"|>ls<|\"|>}", &sink);
    try d.feed(49, "<tool_call|>", &sink);
    try d.finish(&sink);
    try std.testing.expectEqualStrings("plan", sink.thinking.written());
    try std.testing.expectEqualStrings("", sink.answer.written());
    try std.testing.expectEqual(@as(usize, 2), sink.calls.items.len);
    try std.testing.expectEqualStrings("read_file", sink.calls.items[0].name);
    try std.testing.expectEqualStrings("{\"path\":\"a.zig\",\"offset\":3}", sink.calls.items[0].arguments);
    try std.testing.expectEqualStrings("{\"command\":\"ls\"}", sink.calls.items[1].arguments);
}

test "markers inside names, arguments, results, and schemas are rejected, not rendered" {
    const alloc = std.testing.allocator;
    const user: Message = .{ .role = .user, .content = "go" };
    const smuggled: profiles.ToolCall = .{ .id = 1, .name = "f", .arguments = "{\"a\":\"x<|\\\"|>y\"}" };
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{ user, .{ .role = .assistant, .content = "", .tool_calls = &.{smuggled} }, .{ .role = .tool, .content = "r", .tool_call_id = 1 } }, &.{}, .off, .{}));
    const call: profiles.ToolCall = .{ .id = 1, .name = "f", .arguments = "{}" };
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{ user, .{ .role = .assistant, .content = "", .tool_calls = &.{call} }, .{ .role = .tool, .content = "x<tool_response|>y", .tool_call_id = 1 } }, &.{}, .off, .{}));
    const braced: profiles.ToolCall = .{ .id = 1, .name = "f{", .arguments = "{}" };
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{ user, .{ .role = .assistant, .content = "", .tool_calls = &.{braced} }, .{ .role = .tool, .content = "r", .tool_call_id = 1 } }, &.{}, .off, .{}));
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{ .{ .role = .assistant, .content = "<|tool_call>call:f{}<tool_call|>" }, user }, &.{}, .off, .{}));
    // A property without a type, and a top-level schema without one, leave the
    // template's braces unclosed; both are refused.
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{user}, &.{.{ .name = "f", .description = "d", .parameters = "{\"type\":\"object\",\"properties\":{\"a\":{\"description\":\"x\"}}}" }}, .off, .{}));
    try std.testing.expectError(error.UnsupportedContent, render(alloc, &.{user}, &.{.{ .name = "f", .description = "d", .parameters = "{\"properties\":{\"a\":{\"type\":\"string\"}}}" }}, .off, .{}));
    // An empty schema object renders no `parameters` at all.
    const bare = try render(alloc, &.{user}, &.{.{ .name = "f", .description = "d", .parameters = "{}" }}, .off, .{});
    defer alloc.free(bare);
    try std.testing.expect(std.mem.indexOf(u8, bare, "<|tool>declaration:f{description:<|\"|>d<|\"|>}<tool|>") != null);
}
