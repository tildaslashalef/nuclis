//! `POST /v1/chat/completions` on the wire, without a model: a request read
//! into the profile's shapes (messages, tools, images still encoded, effort,
//! sampling, budget), and a completion's events collected and written back
//! in OpenAI's response shape. Every refusal names the field at fault.
//! What a field means and why some are refused: docs/guide/api.md § Chat Completions.
const std = @import("std");
const inference = @import("inference");
const data_url = @import("../../data_url.zig");

const Allocator = std.mem.Allocator;
const Profile = inference.profiles;
const Value = std.json.Value;

/// Output tokens a request may ask for; a larger `max_tokens` is capped, not
/// refused, since clients size it from their own idea of the window.
pub const max_output_tokens = 16384;
pub const max_images = inference.decide.max_images;
pub const max_image_bytes = inference.decide.max_image_bytes;

pub const Request = struct {
    model: ?[]const u8 = null,
    /// Images are not yet on the messages: `image_counts[i]` of `images`
    /// belong to message `i`, in order; the worker encodes them.
    messages: []Profile.Message,
    images: []const []const u8,
    image_counts: []const usize,
    tools: []const Profile.ToolDefinition,
    effort: ?Profile.Effort = null,
    sampling: inference.sampling.Overrides = .{},
    seed: ?u64 = null,
    max_tokens: ?usize = null,
};

/// Why a request was refused: a 400 with this code, message, and field.
pub const Problem = struct {
    code: []const u8 = "invalid_request",
    message: []const u8 = "",
    param: ?[]const u8 = null,
};

pub const Error = error{ Refused, OutOfMemory };

const Reader = struct {
    arena: Allocator,
    problem: *Problem,
    /// Host ids for the conversation's call ids, by first appearance (id = index + 1).
    call_ids: std.ArrayList([]const u8) = .empty,
    images: std.ArrayList([]const u8) = .empty,

    fn refuse(self: *Reader, code: []const u8, param: ?[]const u8, comptime format: []const u8, args: anytype) Error {
        self.problem.* = .{ .code = code, .param = param, .message = try self.arena.print(format, args) };
        return error.Refused;
    }

    fn invalid(self: *Reader, param: []const u8, comptime format: []const u8, args: anytype) Error {
        return self.refuse("invalid_request", param, format, args);
    }

    fn unsupported(self: *Reader, param: []const u8, comptime format: []const u8, args: anytype) Error {
        return self.refuse("unsupported_feature", param, format, args);
    }

    fn callId(self: *Reader, text: []const u8) Error!u32 {
        for (self.call_ids.items, 1..) |known, id| if (std.mem.eql(u8, known, text)) return @intCast(id);
        try self.call_ids.append(self.arena, text);
        return @intCast(self.call_ids.items.len);
    }

    fn knownCallId(self: *Reader, text: []const u8) ?u32 {
        for (self.call_ids.items, 1..) |known, id| if (std.mem.eql(u8, known, text)) return @intCast(id);
        return null;
    }
};

/// Reads `body`. `Refused` sets `problem`; every slice lives in `arena`.
pub fn parse(arena: Allocator, body: []const u8, problem: *Problem) Error!Request {
    var r: Reader = .{ .arena = arena, .problem = problem };
    const root = std.json.parseFromSliceLeaky(Value, arena, body, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return r.refuse("invalid_json", null, "the request body is not valid JSON ({s})", .{@errorName(err)}),
    };
    if (root != .object) return r.invalid("body", "the request body must be a JSON object", .{});
    const o = root.object;
    try refuseUnsupported(&r, o);

    var request: Request = .{ .messages = &.{}, .images = &.{}, .image_counts = &.{}, .tools = &.{} };
    if (o.get("model")) |m| switch (m) {
        .string => |s| request.model = s,
        .null => {},
        else => return r.invalid("model", "\"model\" must be a string", .{}),
    };
    const list = switch (o.get("messages") orelse return r.invalid("messages", "\"messages\" is required", .{})) {
        .array => |a| a.items,
        else => return r.invalid("messages", "\"messages\" must be a list", .{}),
    };
    if (list.len == 0) return r.invalid("messages", "\"messages\" is empty", .{});
    const limits: Profile.Limits = .{};
    if (list.len > limits.messages) return r.invalid("messages", "at most {d} messages; this request has {d}", .{ limits.messages, list.len });
    const messages = try arena.alloc(Profile.Message, list.len);
    const counts = try arena.alloc(usize, list.len);
    for (list, messages, counts, 0..) |item, *message, *count, i| {
        const before = r.images.items.len;
        message.* = try readMessage(&r, item, i);
        count.* = r.images.items.len - before;
    }
    request.messages = messages;
    request.image_counts = counts;
    request.images = r.images.items;

    const tool_choice: enum { auto, none } = if (o.get("tool_choice")) |c| switch (c) {
        .null => .auto,
        .string => |s| if (std.mem.eql(u8, s, "auto")) .auto else if (std.mem.eql(u8, s, "none")) .none else if (std.mem.eql(u8, s, "required"))
            return r.unsupported("tool_choice", "\"tool_choice\": \"required\" needs constrained decoding, which nuclis does not have; use \"auto\"", .{})
        else
            return r.invalid("tool_choice", "\"tool_choice\" is \"auto\", \"none\", or \"required\"", .{}),
        .object => return r.unsupported("tool_choice", "a named \"tool_choice\" needs constrained decoding, which nuclis does not have; use \"auto\"", .{}),
        else => return r.invalid("tool_choice", "\"tool_choice\" is \"auto\" or \"none\"", .{}),
    } else .auto;
    if (o.get("tools")) |t| {
        if (tool_choice == .auto) request.tools = try readTools(&r, t, limits.tools);
    }

    if (o.get("reasoning_effort")) |e| switch (e) {
        .null => {},
        .string => |s| request.effort = effortOf(s) orelse
            return r.invalid("reasoning_effort", "\"reasoning_effort\" is none, minimal, low, medium, high, or xhigh, not \"{s}\"", .{s}),
        else => return r.invalid("reasoning_effort", "\"reasoning_effort\" must be a string", .{}),
    };
    request.sampling = .{
        .temperature = try float(&r, o, "temperature", 0, 2),
        .top_p = try float(&r, o, "top_p", 0, 1),
        .min_p = try float(&r, o, "min_p", 0, 1),
        .presence_penalty = try float(&r, o, "presence_penalty", -2, 2),
        .repetition_penalty = try float(&r, o, "repetition_penalty", 0, 10),
        .top_k = try integer(&r, o, "top_k", 0, std.math.maxInt(u32)),
    };
    if (request.sampling.top_p) |p| if (p == 0) return r.invalid("top_p", "\"top_p\" must be above 0", .{});
    if (request.sampling.repetition_penalty) |p| if (p == 0) return r.invalid("repetition_penalty", "\"repetition_penalty\" must be above 0", .{});
    if (o.get("seed")) |s| switch (s) {
        .null => {},
        .integer => |n| request.seed = @bitCast(n),
        else => return r.invalid("seed", "\"seed\" must be an integer", .{}),
    };
    const budget = try integer(&r, o, "max_completion_tokens", 1, std.math.maxInt(u32)) orelse
        try integer(&r, o, "max_tokens", 1, std.math.maxInt(u32));
    if (budget) |b| request.max_tokens = @min(b, max_output_tokens);
    return request;
}

/// Fields that would change the result and that nuclis cannot honour.
/// Fields that are only advice (`user`, `store`, `metadata`,
/// `prompt_cache_key`, `service_tier`, `parallel_tool_calls`,
/// `stream_options`) are accepted and ignored.
fn refuseUnsupported(r: *Reader, o: std.json.ObjectMap) Error!void {
    if (o.get("stream")) |s| if (s == .bool and s.bool)
        return r.unsupported("stream", "streamed responses are not served yet; send \"stream\": false", .{});
    if (o.get("n")) |n| if (!(n == .null or (n == .integer and n.integer == 1)))
        return r.unsupported("n", "one choice per request (\"n\": 1)", .{});
    if (o.get("logprobs")) |l| if (l == .bool and l.bool)
        return r.unsupported("logprobs", "log probabilities are not returned", .{});
    if (o.get("top_logprobs")) |l| if (!(l == .null or (l == .integer and l.integer == 0)))
        return r.unsupported("top_logprobs", "log probabilities are not returned", .{});
    if (o.get("response_format")) |f| if (f == .object) if (f.object.get("type")) |t| if (t == .string and !std.mem.eql(u8, t.string, "text"))
        return r.unsupported("response_format", "\"response_format\" {s} needs constrained decoding, which nuclis does not have", .{t.string});
    if (o.get("stop")) |s| if (!(s == .null or (s == .array and s.array.items.len == 0)))
        return r.unsupported("stop", "custom stop sequences are not supported; the model's own end of turn stops it", .{});
    if (o.get("logit_bias")) |b| if (b == .object and b.object.count() > 0)
        return r.unsupported("logit_bias", "\"logit_bias\" is not supported", .{});
    if (o.get("frequency_penalty")) |f| if (!(f == .null or (f == .integer and f.integer == 0) or (f == .float and f.float == 0)))
        return r.unsupported("frequency_penalty", "\"frequency_penalty\" is not supported; \"presence_penalty\" and \"repetition_penalty\" are", .{});
    if (o.get("audio") != null or o.get("prediction") != null)
        return r.unsupported(if (o.get("audio") != null) "audio" else "prediction", "audio output and predicted outputs are not supported", .{});
    if (o.get("modalities")) |m| if (m == .array) for (m.array.items) |item| if (item == .string and !std.mem.eql(u8, item.string, "text"))
        return r.unsupported("modalities", "text is the only output modality", .{});
}

fn readMessage(r: *Reader, item: Value, index: usize) Error!Profile.Message {
    const param = try r.arena.print("messages[{d}]", .{index});
    if (item != .object) return r.invalid(param, "{s} must be an object", .{param});
    const o = item.object;
    const role_text = switch (o.get("role") orelse return r.invalid(param, "{s} has no \"role\"", .{param})) {
        .string => |s| s,
        else => return r.invalid(param, "{s}.role must be a string", .{param}),
    };
    const role = std.meta.stringToEnum(Profile.Role, role_text) orelse
        return r.invalid(param, "{s}.role is system, developer, user, assistant, or tool, not \"{s}\"", .{ param, role_text });
    var message: Profile.Message = .{ .role = role, .content = try readContent(r, o.get("content"), role, param) };
    switch (role) {
        .assistant => {
            for ([_][]const u8{ "reasoning_content", "reasoning" }) |field| {
                const v = o.get(field) orelse continue;
                if (v != .string) continue;
                message.reasoning_content = v.string;
                break;
            }
            if (o.get("tool_calls")) |calls| message.tool_calls = try readCalls(r, calls, param);
        },
        .tool => {
            const id = switch (o.get("tool_call_id") orelse return r.invalid(param, "{s} (a tool result) has no \"tool_call_id\"", .{param})) {
                .string => |s| s,
                else => return r.invalid(param, "{s}.tool_call_id must be a string", .{param}),
            };
            message.tool_call_id = r.knownCallId(id) orelse
                return r.invalid(param, "{s}.tool_call_id \"{s}\" answers no earlier assistant call", .{ param, id });
        },
        else => {},
    }
    return message;
}

/// A message's text: a string, null (an assistant turn of calls only), or
/// parts. Image parts are kept, encoded, in `r.images`; only user messages
/// carry them.
fn readContent(r: *Reader, value: ?Value, role: Profile.Role, param: []const u8) Error![]const u8 {
    const v = value orelse return "";
    switch (v) {
        .null => return "",
        .string => |s| return s,
        .array => |parts| {
            var text: std.ArrayList(u8) = .empty;
            for (parts.items, 0..) |part, j| {
                const part_param = try r.arena.print("{s}.content[{d}]", .{ param, j });
                if (part != .object) return r.invalid(part_param, "{s} must be an object", .{part_param});
                const kind = switch (part.object.get("type") orelse .null) {
                    .string => |s| s,
                    else => return r.invalid(part_param, "{s} has no \"type\"", .{part_param}),
                };
                if (std.mem.eql(u8, kind, "text")) {
                    const t = part.object.get("text") orelse .null;
                    if (t != .string) return r.invalid(part_param, "{s}.text must be a string", .{part_param});
                    if (text.items.len > 0 and t.string.len > 0) try text.append(r.arena, '\n');
                    try text.appendSlice(r.arena, t.string);
                } else if (std.mem.eql(u8, kind, "image_url")) {
                    if (role != .user) return r.invalid(part_param, "images go on user messages", .{});
                    try readImage(r, part.object.get("image_url") orelse .null, part_param);
                } else {
                    return r.unsupported(part_param, "{s} is a \"{s}\" part; text and image_url parts are supported", .{ part_param, kind });
                }
            }
            return text.items;
        },
        else => return r.invalid(param, "{s}.content must be a string, null, or a list of parts", .{param}),
    }
}

fn readImage(r: *Reader, value: Value, param: []const u8) Error!void {
    const url = switch (value) {
        .string => |s| s,
        .object => |o| if (o.get("url")) |u| if (u == .string) u.string else null else null,
        else => null,
    } orelse return r.invalid(param, "{s}.image_url must hold a \"url\"", .{param});
    if (!std.mem.startsWith(u8, url, "data:"))
        return r.unsupported(param, "{s}: nuclis fetches no URL; send the image as a base64 data URL", .{param});
    if (r.images.items.len == max_images) return r.invalid(param, "at most {d} images per request", .{max_images});
    const encoded = data_url.payload(url) orelse return r.invalid(param, "{s}: a data URL must be base64 (data:<type>;base64,…)", .{param});
    const bytes = data_url.decode(r.arena, encoded, max_image_bytes) catch |err| switch (err) {
        error.NotBase64 => return r.invalid(param, "{s}: not base64", .{param}),
        error.TooLarge => return r.invalid(param, "{s}: an image is at most {d} bytes", .{ param, max_image_bytes }),
        error.OutOfMemory => return error.OutOfMemory,
    };
    try r.images.append(r.arena, bytes);
}

fn readCalls(r: *Reader, value: Value, param: []const u8) Error![]const Profile.ToolCall {
    const items = switch (value) {
        .null => return &.{},
        .array => |a| a.items,
        else => return r.invalid(param, "{s}.tool_calls must be a list", .{param}),
    };
    const calls = try r.arena.alloc(Profile.ToolCall, items.len);
    for (items, calls, 0..) |item, *call, j| {
        const call_param = try r.arena.print("{s}.tool_calls[{d}]", .{ param, j });
        if (item != .object) return r.invalid(call_param, "{s} must be an object", .{call_param});
        const id = switch (item.object.get("id") orelse .null) {
            .string => |s| s,
            else => return r.invalid(call_param, "{s}.id must be a string", .{call_param}),
        };
        const function = switch (item.object.get("function") orelse .null) {
            .object => |f| f,
            else => return r.invalid(call_param, "{s}.function must be an object", .{call_param}),
        };
        const name = switch (function.get("name") orelse .null) {
            .string => |s| s,
            else => return r.invalid(call_param, "{s}.function.name must be a string", .{call_param}),
        };
        call.* = .{ .id = try r.callId(id), .name = name, .arguments = try arguments(r, function.get("arguments") orelse .null, call_param) };
    }
    return calls;
}

/// A call's arguments as a JSON object's text: a string holding one (empty
/// is `{}`), or the object itself.
fn arguments(r: *Reader, value: Value, param: []const u8) Error![]const u8 {
    switch (value) {
        .null => return "{}",
        .object => return std.json.Stringify.valueAlloc(r.arena, value, .{}),
        .string => |s| {
            if (std.mem.trim(u8, s, " \t\r\n").len == 0) return "{}";
            const parsed = std.json.parseFromSliceLeaky(Value, r.arena, s, .{}) catch
                return r.invalid(param, "{s}.function.arguments is not JSON", .{param});
            if (parsed != .object) return r.invalid(param, "{s}.function.arguments must be a JSON object", .{param});
            return s;
        },
        else => return r.invalid(param, "{s}.function.arguments must be a string of JSON", .{param}),
    }
}

fn readTools(r: *Reader, value: Value, limit: usize) Error![]const Profile.ToolDefinition {
    const items = switch (value) {
        .null => return &.{},
        .array => |a| a.items,
        else => return r.invalid("tools", "\"tools\" must be a list", .{}),
    };
    if (items.len > limit) return r.invalid("tools", "at most {d} tools; this request has {d}", .{ limit, items.len });
    const tools = try r.arena.alloc(Profile.ToolDefinition, items.len);
    for (items, tools, 0..) |item, *tool, i| {
        const param = try r.arena.print("tools[{d}]", .{i});
        if (item != .object) return r.invalid(param, "{s} must be an object", .{param});
        const kind = item.object.get("type") orelse .null;
        if (kind != .string or !std.mem.eql(u8, kind.string, "function"))
            return r.unsupported(param, "{s}: only \"function\" tools are supported", .{param});
        const function = switch (item.object.get("function") orelse .null) {
            .object => |f| f,
            else => return r.invalid(param, "{s}.function must be an object", .{param}),
        };
        const name = switch (function.get("name") orelse .null) {
            .string => |s| s,
            else => return r.invalid(param, "{s}.function.name must be a string", .{param}),
        };
        const description = switch (function.get("description") orelse .null) {
            .string => |s| s,
            .null => "",
            else => return r.invalid(param, "{s}.function.description must be a string", .{param}),
        };
        const parameters = switch (function.get("parameters") orelse .null) {
            .null => "{\"type\":\"object\",\"properties\":{}}",
            .object => try std.json.Stringify.valueAlloc(r.arena, function.get("parameters").?, .{}),
            else => return r.invalid(param, "{s}.function.parameters must be a JSON schema object", .{param}),
        };
        tool.* = .{ .name = name, .description = description, .parameters = parameters };
    }
    return tools;
}

/// OpenAI's effort names onto the profile's: `none` is `off`, `minimal` is
/// `low`, `max` is `xhigh`; the profile's own names pass.
pub fn effortOf(text: []const u8) ?Profile.Effort {
    if (std.mem.eql(u8, text, "none")) return .off;
    if (std.mem.eql(u8, text, "minimal")) return .low;
    if (std.mem.eql(u8, text, "max")) return .xhigh;
    return std.meta.stringToEnum(Profile.Effort, text);
}

fn number(value: Value) ?f64 {
    return switch (value) {
        .integer => |n| @floatFromInt(n),
        .float => |f| f,
        else => null,
    };
}

fn float(r: *Reader, o: std.json.ObjectMap, field: []const u8, low: f32, high: f32) Error!?f32 {
    const value = o.get(field) orelse return null;
    if (value == .null) return null;
    const n = number(value) orelse return r.invalid(field, "\"{s}\" must be a number", .{field});
    if (!(n >= low and n <= high)) return r.invalid(field, "\"{s}\" must be from {d} to {d}", .{ field, low, high });
    return @floatCast(n);
}

fn integer(r: *Reader, o: std.json.ObjectMap, field: []const u8, low: i64, high: i64) Error!?usize {
    const value = o.get(field) orelse return null;
    switch (value) {
        .null => return null,
        .integer => |n| {
            if (n < low or n > high) return r.invalid(field, "\"{s}\" must be from {d} to {d}", .{ field, low, high });
            return @intCast(n);
        },
        else => return r.invalid(field, "\"{s}\" must be an integer", .{field}),
    }
}

// ----- the response -----

/// One completion's events, copied out of the sink as they arrive.
pub const Collector = struct {
    arena: Allocator,
    reasoning: std.ArrayList(u8) = .empty,
    content: std.ArrayList(u8) = .empty,
    calls: std.ArrayList(Call) = .empty,

    pub const Call = struct { name: []const u8, arguments: []const u8 };

    pub fn send(self: *Collector, event: inference.events.Event) Allocator.Error!void {
        switch (event) {
            .thinking => |text| try self.reasoning.appendSlice(self.arena, text),
            .answer => |text| try self.content.appendSlice(self.arena, text),
            .tool_call => |call| try self.calls.append(self.arena, .{
                .name = try self.arena.dupe(u8, call.name),
                .arguments = try self.arena.dupe(u8, call.arguments),
            }),
            .tool_progress, .tool_cut, .stop => {},
        }
    }
};

pub const FinishReason = enum { stop, length, tool_calls };

pub fn finishReason(stop: inference.engine.StopReason, calls: usize) FinishReason {
    if (calls > 0) return .tool_calls;
    return switch (stop) {
        .token_budget, .context_limit => .length,
        .eos, .cancelled, .failure => .stop,
    };
}

pub const Usage = struct {
    prompt: usize,
    cached: usize,
    completion: usize,
    reasoning: usize,
};

/// What the response says besides the collected text.
pub const Meta = struct {
    id: []const u8,
    created: i64,
    model: []const u8,
    finish: FinishReason,
    usage: Usage,
    /// One per collected call, in order.
    call_ids: []const []const u8,
};

/// The `chat.completion` object.
pub fn writeResponse(out: *std.Io.Writer, arena: Allocator, collected: *const Collector, meta: Meta) !void {
    std.debug.assert(meta.call_ids.len == collected.calls.items.len);
    var s: std.json.Stringify = .{ .writer = out, .options = .{ .whitespace = .indent_2 } };
    try s.beginObject();
    try s.objectField("id");
    try s.write(meta.id);
    try s.objectField("object");
    try s.write("chat.completion");
    try s.objectField("created");
    try s.write(meta.created);
    try s.objectField("model");
    try s.write(meta.model);
    try s.objectField("choices");
    try s.beginArray();
    try s.beginObject();
    try s.objectField("index");
    try s.write(0);
    try s.objectField("message");
    try s.beginObject();
    try s.objectField("role");
    try s.write("assistant");
    try s.objectField("content");
    const content = try validUtf8(arena, collected.content.items);
    if (content.len == 0 and collected.calls.items.len > 0) try s.write(null) else try s.write(content);
    if (collected.reasoning.items.len > 0) {
        try s.objectField("reasoning_content");
        try s.write(try validUtf8(arena, collected.reasoning.items));
    }
    if (collected.calls.items.len > 0) {
        try s.objectField("tool_calls");
        try s.beginArray();
        for (collected.calls.items, meta.call_ids) |call, id| {
            try s.beginObject();
            try s.objectField("id");
            try s.write(id);
            try s.objectField("type");
            try s.write("function");
            try s.objectField("function");
            try s.beginObject();
            try s.objectField("name");
            try s.write(try validUtf8(arena, call.name));
            try s.objectField("arguments");
            try s.write(try validUtf8(arena, call.arguments));
            try s.endObject();
            try s.endObject();
        }
        try s.endArray();
    }
    try s.endObject();
    try s.objectField("logprobs");
    try s.write(null);
    try s.objectField("finish_reason");
    try s.write(@tagName(meta.finish));
    try s.endObject();
    try s.endArray();
    try s.objectField("usage");
    try s.beginObject();
    try s.objectField("prompt_tokens");
    try s.write(meta.usage.prompt);
    try s.objectField("completion_tokens");
    try s.write(meta.usage.completion);
    try s.objectField("total_tokens");
    try s.write(meta.usage.prompt + meta.usage.completion);
    try s.objectField("prompt_tokens_details");
    try s.beginObject();
    try s.objectField("cached_tokens");
    try s.write(meta.usage.cached);
    try s.endObject();
    try s.objectField("completion_tokens_details");
    try s.beginObject();
    try s.objectField("reasoning_tokens");
    try s.write(meta.usage.reasoning);
    try s.endObject();
    try s.endObject();
    try s.endObject();
    try out.writeByte('\n');
}

/// `text` with every invalid UTF-8 sequence replaced by U+FFFD: model text
/// is decoded token by token and is not trusted to be well formed.
pub fn validUtf8(arena: Allocator, text: []const u8) Allocator.Error![]const u8 {
    if (std.unicode.utf8ValidateSlice(text)) return text;
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < text.len) {
        const length = std.unicode.utf8ByteSequenceLength(text[i]) catch 0;
        if (length > 0 and i + length <= text.len and std.unicode.utf8ValidateSlice(text[i .. i + length])) {
            try out.appendSlice(arena, text[i .. i + length]);
            i += length;
        } else {
            // One replacement for a truncated sequence: its lead byte and
            // the continuation bytes that follow it.
            try out.appendSlice(arena, "\u{FFFD}");
            i += 1;
            var left = if (length > 1) length - 1 else 0;
            while (left > 0 and i < text.len and text[i] & 0xC0 == 0x80) : (left -= 1) i += 1;
        }
    }
    return out.items;
}

/// `prefix` followed by 24 random base62 characters.
pub fn newId(arena: Allocator, io: std.Io, prefix: []const u8) Allocator.Error![]const u8 {
    const alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    var raw: [24]u8 = undefined;
    io.randomSecure(&raw) catch io.random(&raw);
    const out = try arena.alloc(u8, prefix.len + raw.len);
    @memcpy(out[0..prefix.len], prefix);
    for (raw, out[prefix.len..]) |byte, *c| c.* = alphabet[byte % alphabet.len];
    return out;
}

// ----- tests -----

const testing = std.testing;

fn parseOk(arena: Allocator, body: []const u8) !Request {
    var problem: Problem = .{};
    return parse(arena, body, &problem) catch |err| {
        std.debug.print("refused: {s} {s} ({?s})\n", .{ problem.code, problem.message, problem.param });
        return err;
    };
}

test "a conversation as an agent sends it: roles, an image, calls and their result, tools" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const request = try parseOk(arena,
        \\{"model": "qwen3.8-27b", "stream": false, "store": false, "stream_options": {"include_usage": true},
        \\ "messages": [
        \\  {"role": "developer", "content": "You are terse."},
        \\  {"role": "user", "content": [{"type": "text", "text": "What is in"}, {"type": "text", "text": "this?"},
        \\     {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}}]},
        \\  {"role": "assistant", "content": null, "reasoning_content": "Look first.",
        \\     "tool_calls": [{"id": "call_a|fc_1", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"x\"}"}},
        \\                    {"id": "b", "type": "function", "function": {"name": "ls", "arguments": ""}}]},
        \\  {"role": "tool", "tool_call_id": "call_a|fc_1", "content": "hello"},
        \\  {"role": "tool", "tool_call_id": "b", "content": [{"type": "text", "text": "(no tool output)"}]}],
        \\ "tools": [{"type": "function", "function": {"name": "read", "description": "Read a file.",
        \\            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
        \\           {"type": "function", "function": {"name": "ls"}}],
        \\ "reasoning_effort": "minimal", "max_completion_tokens": 120000, "temperature": 0.6, "top_k": 20, "seed": 7}
    );
    try testing.expectEqualStrings("qwen3.8-27b", request.model.?);
    try testing.expectEqual(@as(usize, 5), request.messages.len);
    try testing.expectEqual(Profile.Role.developer, request.messages[0].role);
    try testing.expectEqualStrings("What is in\nthis?", request.messages[1].content);
    try testing.expectEqualSlices(usize, &.{ 0, 1, 0, 0, 0 }, request.image_counts);
    try testing.expectEqualStrings("hi", request.images[0]);
    const assistant = request.messages[2];
    try testing.expectEqualStrings("", assistant.content);
    try testing.expectEqualStrings("Look first.", assistant.reasoning_content);
    try testing.expectEqual(@as(u32, 1), assistant.tool_calls[0].id);
    try testing.expectEqual(@as(u32, 2), assistant.tool_calls[1].id);
    try testing.expectEqualStrings("{}", assistant.tool_calls[1].arguments);
    try testing.expectEqual(@as(?u32, 1), request.messages[3].tool_call_id);
    try testing.expectEqual(@as(?u32, 2), request.messages[4].tool_call_id);
    try testing.expectEqualStrings("(no tool output)", request.messages[4].content);
    try testing.expectEqual(@as(usize, 2), request.tools.len);
    try testing.expectEqualStrings("{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}}}", request.tools[0].parameters);
    try testing.expectEqualStrings("{\"type\":\"object\",\"properties\":{}}", request.tools[1].parameters);
    try testing.expectEqual(@as(?Profile.Effort, .low), request.effort);
    try testing.expectEqual(@as(?usize, max_output_tokens), request.max_tokens);
    try testing.expectEqual(@as(?f32, 0.6), request.sampling.temperature);
    try testing.expectEqual(@as(?usize, 20), request.sampling.top_k);
    try testing.expectEqual(@as(?u64, 7), request.seed);
}

test "tool_choice none drops the tools; max_tokens is read when max_completion_tokens is absent" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const request = try parseOk(arena_state.allocator(),
        \\{"messages": [{"role": "user", "content": "hi"}], "tool_choice": "none", "max_tokens": 64,
        \\ "tools": [{"type": "function", "function": {"name": "ls"}}]}
    );
    try testing.expectEqual(@as(usize, 0), request.tools.len);
    try testing.expectEqual(@as(?usize, 64), request.max_tokens);
    try testing.expect(request.model == null and request.effort == null);
}

test "refusals name their code and field" {
    const cases = [_]struct { body: []const u8, code: []const u8, param: ?[]const u8 }{
        .{ .body = "{", .code = "invalid_json", .param = null },
        .{ .body = "[]", .code = "invalid_request", .param = "body" },
        .{ .body = "{\"messages\": []}", .code = "invalid_request", .param = "messages" },
        .{ .body = "{\"messages\": [{\"role\": \"function\", \"content\": \"x\"}]}", .code = "invalid_request", .param = "messages[0]" },
        .{ .body = "{\"messages\": [{\"role\": \"user\", \"content\": \"x\"}], \"stream\": true}", .code = "unsupported_feature", .param = "stream" },
        .{ .body = "{\"messages\": [{\"role\": \"user\", \"content\": \"x\"}], \"n\": 2}", .code = "unsupported_feature", .param = "n" },
        .{ .body = "{\"messages\": [{\"role\": \"user\", \"content\": \"x\"}], \"response_format\": {\"type\": \"json_schema\"}}", .code = "unsupported_feature", .param = "response_format" },
        .{ .body = "{\"messages\": [{\"role\": \"user\", \"content\": \"x\"}], \"tool_choice\": \"required\"}", .code = "unsupported_feature", .param = "tool_choice" },
        .{ .body = "{\"messages\": [{\"role\": \"user\", \"content\": \"x\"}], \"tool_choice\": {\"type\": \"function\"}}", .code = "unsupported_feature", .param = "tool_choice" },
        .{ .body = "{\"messages\": [{\"role\": \"user\", \"content\": \"x\"}], \"stop\": [\"\\n\"]}", .code = "unsupported_feature", .param = "stop" },
        .{ .body = "{\"messages\": [{\"role\": \"user\", \"content\": \"x\"}], \"reasoning_effort\": \"extreme\"}", .code = "invalid_request", .param = "reasoning_effort" },
        .{ .body = "{\"messages\": [{\"role\": \"user\", \"content\": \"x\"}], \"temperature\": 3}", .code = "invalid_request", .param = "temperature" },
        .{ .body = "{\"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"image_url\", \"image_url\": {\"url\": \"https://x/y.png\"}}]}]}", .code = "unsupported_feature", .param = "messages[0].content[0]" },
        .{ .body = "{\"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"input_audio\"}]}]}", .code = "unsupported_feature", .param = "messages[0].content[0]" },
        .{ .body = "{\"messages\": [{\"role\": \"assistant\", \"content\": [{\"type\": \"image_url\", \"image_url\": \"data:image/png;base64,aGk=\"}]}]}", .code = "invalid_request", .param = "messages[0].content[0]" },
        .{ .body = "{\"messages\": [{\"role\": \"tool\", \"tool_call_id\": \"nope\", \"content\": \"x\"}]}", .code = "invalid_request", .param = "messages[0]" },
        .{ .body = "{\"messages\": [{\"role\": \"assistant\", \"tool_calls\": [{\"id\": \"a\", \"function\": {\"name\": \"f\", \"arguments\": \"[1]\"}}]}]}", .code = "invalid_request", .param = "messages[0].tool_calls[0]" },
        .{ .body = "{\"messages\": [{\"role\": \"user\", \"content\": \"x\"}], \"tools\": [{\"type\": \"web_search\"}]}", .code = "unsupported_feature", .param = "tools[0]" },
    };
    for (cases) |case| {
        errdefer std.debug.print("case: {s}\n", .{case.body});
        var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
        defer arena_state.deinit();
        var problem: Problem = .{};
        try testing.expectError(error.Refused, parse(arena_state.allocator(), case.body, &problem));
        try testing.expectEqualStrings(case.code, problem.code);
        if (case.param) |p| try testing.expectEqualStrings(p, problem.param.?) else try testing.expect(problem.param == null);
        try testing.expect(problem.message.len > 0);
    }
}

test "efforts by OpenAI's names and the profile's" {
    try testing.expectEqual(@as(?Profile.Effort, .off), effortOf("none"));
    try testing.expectEqual(@as(?Profile.Effort, .low), effortOf("minimal"));
    try testing.expectEqual(@as(?Profile.Effort, .high), effortOf("high"));
    try testing.expectEqual(@as(?Profile.Effort, .xhigh), effortOf("max"));
    try testing.expectEqual(@as(?Profile.Effort, .off), effortOf("off"));
    try testing.expect(effortOf("auto") == null);
}

test "the response: reasoning, calls with their ids, null content beside calls, usage" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var collected: Collector = .{ .arena = arena };
    try collected.send(.{ .thinking = "Need the file." });
    try collected.send(.{ .tool_progress = 3 });
    try collected.send(.{ .tool_call = .{ .id = 1, .name = "read", .arguments = "{\"path\":\"x\"}" } });
    const finish = finishReason(.eos, collected.calls.items.len);
    try testing.expectEqual(FinishReason.tool_calls, finish);
    var out: std.Io.Writer.Allocating = .init(arena);
    try writeResponse(&out.writer, arena, &collected, .{
        .id = "chatcmpl-x",
        .created = 1,
        .model = "m",
        .finish = finish,
        .usage = .{ .prompt = 100, .cached = 80, .completion = 12, .reasoning = 5 },
        .call_ids = &.{"call_1"},
    });
    const parsed = try std.json.parseFromSliceLeaky(Value, arena, out.written(), .{});
    const choice = parsed.object.get("choices").?.array.items[0].object;
    const message = choice.get("message").?.object;
    try testing.expect(message.get("content").? == .null);
    try testing.expectEqualStrings("Need the file.", message.get("reasoning_content").?.string);
    const call = message.get("tool_calls").?.array.items[0].object;
    try testing.expectEqualStrings("call_1", call.get("id").?.string);
    try testing.expectEqualStrings("{\"path\":\"x\"}", call.get("function").?.object.get("arguments").?.string);
    try testing.expectEqualStrings("tool_calls", choice.get("finish_reason").?.string);
    const usage = parsed.object.get("usage").?.object;
    try testing.expectEqual(@as(i64, 112), usage.get("total_tokens").?.integer);
    try testing.expectEqual(@as(i64, 80), usage.get("prompt_tokens_details").?.object.get("cached_tokens").?.integer);
    try testing.expectEqual(@as(i64, 5), usage.get("completion_tokens_details").?.object.get("reasoning_tokens").?.integer);
    try testing.expectEqual(FinishReason.length, finishReason(.token_budget, 0));
    try testing.expectEqual(FinishReason.stop, finishReason(.eos, 0));
}

test "invalid UTF-8 from the model becomes U+FFFD; ids are the prefix and 24 base62 characters" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try testing.expectEqualStrings("ok", try validUtf8(arena, "ok"));
    try testing.expectEqualStrings("a\u{FFFD}b\u{FFFD}", try validUtf8(arena, "a\xffb\xe2\x82"));
    const id = try newId(arena, testing.io, "call_");
    try testing.expectEqual(@as(usize, 29), id.len);
    try testing.expect(std.mem.startsWith(u8, id, "call_"));
}
