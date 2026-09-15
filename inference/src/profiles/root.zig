//! Checkpoint prompt conventions are independent of numerical architectures.
//!
//! This root is the profile registry. The conversation types every
//! profile renders (`Role`, `Message`, `Effort`, `Limits`) are shared so
//! the executable's configuration and chat speak one vocabulary; each
//! profile decides what a value means for its template (Qwen3.8 has four
//! reasoning efforts, a profile whose template only switches thinking on
//! or off maps `off` to off and everything else to on). A profile is
//! selected by the SHA-256 of the artifact's `tokenizer.chat_template`,
//! never by architecture: a conversation is never interpreted with a
//! template the profile's fixtures did not verify.
const std = @import("std");
const sampling = @import("../sampling/root.zig");
const gguf = @import("../formats/gguf.zig");
const vocabulary = @import("../tokenizer/vocabulary.zig");
pub const stream = @import("stream.zig");

pub const qwen38 = @import("qwen38.zig");
pub const gemma4 = @import("gemma4.zig");

pub const Role = enum { system, developer, user, assistant, tool };
/// The reasoning control shared by every profile. Named after Qwen3.8's
/// levels because it was the first; profiles with fewer levels collapse them.
pub const Effort = enum { off, low, medium, xhigh };

/// One assistant request to call a tool. `id` is host correlation data: a
/// model's native wire format need not carry it, so the host assigns it and
/// matches the later result back to the call with it. `arguments` is a
/// complete JSON object, normalized from native syntax at the profile
/// boundary; the agent still validates its fields against the tool it names.
pub const ToolCall = struct {
    id: u32,
    name: []const u8,
    arguments: []const u8,
};

/// A tool the model may call, supplied to rendering: the name the model
/// emits, a human-readable description, and its parameter schema as a UTF-8
/// JSON object. All slices are borrowed only for the render call. A profile
/// serializes the definitions into its own tools block; the shared shape
/// carries no wire syntax.
pub const ToolDefinition = struct {
    name: []const u8,
    description: []const u8,
    parameters: []const u8,
};

pub const Message = struct {
    role: Role,
    content: []const u8,
    /// Assistant reasoning, supplied separately and never extracted from
    /// `content`; profiles that keep no reasoning in history ignore it.
    reasoning_content: []const u8 = "",
    /// Calls the assistant requested, in execution order. Nonempty only on
    /// assistant messages; each is answered by exactly one later `.tool`
    /// message carrying the same `id`.
    tool_calls: []const ToolCall = &.{},
    /// For `role = .tool`, the host ID of the assistant call being answered.
    /// Null on every other role.
    tool_call_id: ?u32 = null,
};
pub const Limits = struct {
    messages: usize = 1024,
    input_bytes: usize = 1024 * 1024,
    output_bytes: usize = 2 * 1024 * 1024,
    /// Maximum tool definitions accepted by `render`.
    tools: usize = 64,
};
pub const Error = std.mem.Allocator.Error || error{
    InvalidConversation,
    InvalidUtf8,
    UnsupportedContent,
    ToolsUnsupported,
    LimitExceeded,
};
/// Sampling overrides a caller states per option (command-line flags); null
/// leaves the profile's value in place.
pub const SamplingOverrides = sampling.Overrides;

/// How a checkpoint delimits its reasoning in the text it generates: the
/// marker that opens it (which the model may emit, or the prompt may have
/// already supplied) and the one that closes it. These describe the template;
/// streaming interpretation uses resolved IDs through `Profile.decoder`.
pub const Reasoning = struct { open: []const u8, close: []const u8 };

/// The template's control texts, as a module declares them: reasoning markers,
/// and — when the template defines native tool calling — the call brackets and
/// the body parser. `Profile.decoder` resolves the texts to token IDs.
pub const StreamMarkers = struct {
    open: []const u8,
    close: []const u8,
    open_suffix: []const u8 = "",
    tool_open: ?[]const u8 = null,
    tool_close: ?[]const u8 = null,
    tool_parse: ?stream.ParseTool = null,
};

/// The profiles the tree implements. Each tag names a module with the same
/// surface: `template_sha256`, `render`, `samplingDefaults`, `stop_tokens`,
/// `reasoning`. Add a profile by adding its tag and module here; nothing
/// else in the tree lists them.
pub const Profile = enum {
    qwen38,
    gemma4,

    /// Resolve control tokens against this artifact once per completion.
    /// The profile supplies the non-special header suffix separately.
    pub fn decoder(self: Profile, alloc: std.mem.Allocator, vocab: *const vocabulary.Vocabulary, effort: Effort) !stream.Decoder {
        return switch (self) {
            inline else => |p| blk: {
                const m = p.module().stream_markers;
                var markers: stream.Markers = .{
                    .open = vocab.tokenId(m.open) orelse return error.MissingReasoningToken,
                    .close = vocab.tokenId(m.close) orelse return error.MissingReasoningToken,
                    .open_suffix = m.open_suffix,
                };
                if (m.tool_open) |open_text| {
                    markers.tool = .{
                        .open = vocab.tokenId(open_text) orelse return error.MissingToolToken,
                        .close = vocab.tokenId(m.tool_close.?) orelse return error.MissingToolToken,
                        .parse = m.tool_parse.?,
                    };
                }
                break :blk stream.Decoder.init(alloc, markers, effort != .off);
            },
        };
    }

    /// The module behind a tag, for `inline` dispatch below.
    fn module(comptime self: Profile) type {
        return switch (self) {
            .qwen38 => qwen38,
            .gemma4 => gemma4,
        };
    }

    /// The lowercase hex SHA-256 of the GGUF template the profile implements.
    pub fn templateSha256(self: Profile) []const u8 {
        return switch (self) {
            inline else => |p| p.module().template_sha256,
        };
    }

    /// Renders a completion-ready conversation (ending with a user message or
    /// a completed tool-result group) with optional tool definitions. Caller
    /// owns the result. Profiles that cannot render tools yet return
    /// `error.ToolsUnsupported` for a structurally valid tool input.
    pub fn render(self: Profile, alloc: std.mem.Allocator, messages: []const Message, tools: []const ToolDefinition, effort: Effort, limits: Limits) Error![]u8 {
        return switch (self) {
            inline else => |p| p.module().render(alloc, messages, tools, effort, limits),
        };
    }

    /// The bytes every rendering of a conversation that starts with these
    /// system messages (and tools) begins with, without a generation prompt:
    /// what a session primes with before the first user message. Each
    /// profile pins it as a byte prefix of its `render`.
    pub fn prefix(self: Profile, alloc: std.mem.Allocator, messages: []const Message, tools: []const ToolDefinition, effort: Effort, limits: Limits) Error![]u8 {
        return switch (self) {
            inline else => |p| p.module().prefix(alloc, messages, tools, effort, limits),
        };
    }

    /// The checkpoint's official sampler settings for a reasoning mode.
    pub fn samplingDefaults(self: Profile, effort: Effort) sampling.Options {
        return switch (self) {
            inline else => |p| p.module().samplingDefaults(effort),
        };
    }

    /// The mode's defaults with the caller's overrides applied per option.
    pub fn samplingOptions(self: Profile, effort: Effort, overrides: SamplingOverrides) sampling.Options {
        return self.samplingDefaults(effort).override(overrides);
    }

    /// The texts of the tokens that end a turn under this template; the
    /// engine resolves them to ids in the artifact's vocabulary at load.
    pub fn stopTokens(self: Profile) []const []const u8 {
        return switch (self) {
            inline else => |p| &p.module().stop_tokens,
        };
    }

    /// The reasoning markers of generated text under this template.
    pub fn reasoning(self: Profile) Reasoning {
        return switch (self) {
            inline else => |p| p.module().reasoning,
        };
    }
};

/// The profile pinned to a template digest (lowercase hex), or null.
pub fn forTemplate(sha256_hex: []const u8) ?Profile {
    inline for (comptime std.enums.values(Profile)) |p| {
        if (std.mem.eql(u8, sha256_hex, p.templateSha256())) return p;
    }
    return null;
}

/// The profile for an artifact's `tokenizer.chat_template`, or null when
/// the key is absent or no profile implements that exact template. Reads
/// the header only, so `nuclis tokenize` can ask without building a model.
pub fn forDocument(document: gguf.Document) ?Profile {
    const template = document.string("tokenizer.chat_template") orelse return null;
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(template, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    return forTemplate(&hex);
}

/// The number of leading system/developer messages, which profiles merge or
/// fold into a system turn. `validate` has already rejected a system or
/// developer message after the first non-prefix one.
pub fn leadingSystemCount(messages: []const Message) usize {
    var count: usize = 0;
    for (messages) |message| switch (message.role) {
        .system, .developer => count += 1,
        else => break,
    };
    return count;
}

/// Whether the conversation or its tool definitions use any tool feature a
/// text-only profile cannot render. Does not by itself validate the input.
pub fn usesToolFeatures(messages: []const Message, tools: []const ToolDefinition) bool {
    if (tools.len != 0) return true;
    for (messages) |message| {
        if (message.role == .tool or message.tool_calls.len != 0) return true;
    }
    return false;
}

/// Validates a conversation and its tool definitions without rendering. Every
/// profile calls this before its own template-specific checks, so the shared
/// rules cannot drift: bounds, UTF-8, reasoning placement, the system/developer
/// prefix, and the structure of assistant calls and tool results.
///
/// Tool history is checked as a sequence of *groups*, one per assistant message
/// with calls. Inside a group results must answer the pending calls in
/// execution order, matched by host ID; between groups nothing may interleave.
/// Orphans, duplicates, missing results, and out-of-order matches are rejected.
/// A completion-ready conversation ends with a user message or a fully answered
/// group. The allocator is scratch for JSON validation only; `render` passes
/// its own.
///
/// This validates structure, not support: a structurally valid tool input still
/// returns OK here, and a text-only profile rejects it with
/// `error.ToolsUnsupported` after this call.
pub fn validate(alloc: std.mem.Allocator, messages: []const Message, tools: []const ToolDefinition, limits: Limits) Error!void {
    if (messages.len == 0) return error.InvalidConversation;
    if (messages.len > limits.messages) return error.LimitExceeded;
    if (tools.len > limits.tools) return error.LimitExceeded;

    var input_bytes: usize = 0;
    for (tools, 0..) |tool, i| {
        input_bytes = try countBytes(input_bytes, limits.input_bytes, &.{ tool.name, tool.description, tool.parameters });
        if (tool.name.len == 0) return error.InvalidConversation;
        for (tools[0..i]) |other| {
            if (std.mem.eql(u8, other.name, tool.name)) return error.InvalidConversation;
        }
        if (!std.unicode.utf8ValidateSlice(tool.name) or !std.unicode.utf8ValidateSlice(tool.description)) return error.InvalidUtf8;
        if (!try isJsonObject(alloc, tool.parameters)) return error.InvalidConversation;
    }

    // The calls of the most recent assistant message, and how many of them a
    // `.tool` message has answered so far.
    var pending: []const ToolCall = &.{};
    var answered: usize = 0;
    var in_prefix = true;
    var last: Role = undefined;
    for (messages) |message| {
        input_bytes = try countBytes(input_bytes, limits.input_bytes, &.{ message.content, message.reasoning_content });
        if (!std.unicode.utf8ValidateSlice(message.content) or !std.unicode.utf8ValidateSlice(message.reasoning_content)) return error.InvalidUtf8;
        if (message.role != .assistant and message.reasoning_content.len != 0) return error.UnsupportedContent;
        // System/developer messages may only lead; `in_prefix` flips on the
        // first other role and never returns.
        if (message.role == .system or message.role == .developer) {
            if (!in_prefix) return error.InvalidConversation;
        } else in_prefix = false;

        switch (message.role) {
            .system, .developer => {
                if (message.tool_calls.len != 0 or message.tool_call_id != null) return error.InvalidConversation;
            },
            .user => {
                if (message.tool_calls.len != 0 or message.tool_call_id != null) return error.InvalidConversation;
                if (pending.len != 0) return error.InvalidConversation; // unanswered calls
            },
            .assistant => {
                if (message.tool_call_id != null) return error.InvalidConversation;
                if (pending.len != 0) return error.InvalidConversation; // previous group unanswered
                for (message.tool_calls, 0..) |call, i| {
                    input_bytes = try countBytes(input_bytes, limits.input_bytes, &.{ call.name, call.arguments });
                    if (call.name.len == 0) return error.InvalidConversation;
                    if (!std.unicode.utf8ValidateSlice(call.name)) return error.InvalidUtf8;
                    if (!try isJsonObject(alloc, call.arguments)) return error.InvalidConversation;
                    for (message.tool_calls[0..i]) |other| {
                        if (other.id == call.id) return error.InvalidConversation; // duplicate call ID
                    }
                }
                pending = message.tool_calls;
                answered = 0;
            },
            .tool => {
                if (message.tool_calls.len != 0) return error.InvalidConversation;
                const id = message.tool_call_id orelse return error.InvalidConversation; // orphan result
                if (pending.len == 0 or answered >= pending.len) return error.InvalidConversation; // orphan or duplicate
                if (pending[answered].id != id) return error.InvalidConversation; // out of order
                answered += 1;
                if (answered == pending.len) {
                    pending = &.{};
                    answered = 0;
                }
            },
        }
        last = message.role;
    }
    if (pending.len != 0) return error.InvalidConversation; // missing results
    if (last != .user and last != .tool) return error.InvalidConversation;
}

fn countBytes(total: usize, limit: usize, parts: []const []const u8) Error!usize {
    var used = total;
    for (parts) |part| {
        if (part.len > limit - used) return error.LimitExceeded;
        used += part.len;
    }
    return used;
}

/// Whether `text` is exactly one JSON object (a schema or a call's arguments).
/// Syntax errors and truncation are a structural failure, not an I/O error;
/// deep nesting can still exhaust the scratch allocator.
fn isJsonObject(alloc: std.mem.Allocator, text: []const u8) Error!bool {
    var scanner = std.json.Scanner.initCompleteInput(alloc, text);
    defer scanner.deinit();
    const first = (try nextJson(&scanner)) orelse return false;
    if (first != .object_begin) return false;
    while (true) {
        const token = (try nextJson(&scanner)) orelse return false;
        if (token == .end_of_document) return true;
    }
}

/// The next JSON token, or null for malformed or truncated input. Only a real
/// end-of-document token marks a complete value.
fn nextJson(scanner: *std.json.Scanner) Error!?std.json.Token {
    return scanner.next() catch |err| switch (err) {
        error.SyntaxError, error.UnexpectedEndOfInput => return null,
        error.OutOfMemory => error.OutOfMemory,
        error.BufferUnderrun => unreachable,
    };
}

test "profiles are selected by template digest" {
    try std.testing.expectEqual(Profile.qwen38, forTemplate(qwen38.template_sha256).?);
    try std.testing.expectEqual(Profile.gemma4, forTemplate(gemma4.template_sha256).?);
    try std.testing.expect(forTemplate("") == null);
    try std.testing.expect(forTemplate("0000000000000000000000000000000000000000000000000000000000000000") == null);
    try std.testing.expectEqualStrings(qwen38.template_sha256, Profile.qwen38.templateSha256());
}

test "profile dispatch reaches the module's defaults and renderer" {
    try std.testing.expectEqual(qwen38.samplingDefaults(.off), Profile.qwen38.samplingDefaults(.off));
    try std.testing.expectEqual(qwen38.samplingDefaults(.low).override(.{ .top_k = 3 }), Profile.qwen38.samplingOptions(.low, .{ .top_k = 3 }));
    const text = try Profile.qwen38.render(std.testing.allocator, &.{.{ .role = .user, .content = "hi" }}, &.{}, .off, .{});
    defer std.testing.allocator.free(text);
    const direct = try qwen38.render(std.testing.allocator, &.{.{ .role = .user, .content = "hi" }}, &.{}, .off, .{});
    defer std.testing.allocator.free(direct);
    try std.testing.expectEqualStrings(direct, text);
}

test "every profile names at least one stop token and both reasoning markers" {
    for (std.enums.values(Profile)) |p| {
        try std.testing.expect(p.stopTokens().len >= 1);
        try std.testing.expect(p.reasoning().open.len > 0 and p.reasoning().close.len > 0);
    }
    try std.testing.expectEqualStrings("<|im_end|>", Profile.qwen38.stopTokens()[0]);
    try std.testing.expectEqualStrings("<turn|>", Profile.gemma4.stopTokens()[0]);
    try std.testing.expectEqualStrings("</think>", Profile.qwen38.reasoning().close);
    try std.testing.expectEqualStrings("<channel|>", Profile.gemma4.reasoning().close);
}

const user_message: Message = .{ .role = .user, .content = "hi" };
const read_call: ToolCall = .{ .id = 7, .name = "read_file", .arguments = "{\"path\":\"a\"}" };

test "structurally valid tool history validates; qwen renders it, gemma rejects" {
    const alloc = std.testing.allocator;
    const history = [_]Message{
        .{ .role = .user, .content = "read a" },
        .{ .role = .assistant, .content = "looking", .tool_calls = &.{read_call} },
        .{ .role = .tool, .content = "contents", .tool_call_id = read_call.id },
    };
    try validate(alloc, &history, &.{}, .{});
    try std.testing.expect(usesToolFeatures(&history, &.{}));
    const qwen = try qwen38.render(alloc, &history, &.{}, .off, .{});
    defer alloc.free(qwen);
    try std.testing.expect(std.mem.indexOf(u8, qwen, "<function=read_file>") != null);
    try std.testing.expectError(error.ToolsUnsupported, gemma4.render(alloc, &history, &.{}, .medium, .{}));

    const tool = ToolDefinition{ .name = "read_file", .description = "Read a file.", .parameters = "{\"type\":\"object\"}" };
    try validate(alloc, &.{user_message}, &.{tool}, .{});
    try std.testing.expect(usesToolFeatures(&.{user_message}, &.{tool}));
    const qwen_tools = try qwen38.render(alloc, &.{user_message}, &.{tool}, .off, .{});
    defer alloc.free(qwen_tools);
    try std.testing.expect(std.mem.indexOf(u8, qwen_tools, "<tools>") != null);
    try std.testing.expectError(error.ToolsUnsupported, gemma4.render(alloc, &.{user_message}, &.{tool}, .medium, .{}));

    try std.testing.expect(!usesToolFeatures(&.{user_message}, &.{}));
}

test "tool results answer the pending calls in order and may end a conversation" {
    const alloc = std.testing.allocator;
    const calls = [_]ToolCall{
        .{ .id = 1, .name = "read_file", .arguments = "{}" },
        .{ .id = 2, .name = "glob", .arguments = "{\"pattern\":\"*.zig\"}" },
    };
    const answered = [_]Message{
        .{ .role = .user, .content = "go" },
        .{ .role = .assistant, .content = "thinking", .tool_calls = &calls },
        .{ .role = .tool, .content = "a", .tool_call_id = 1 },
        .{ .role = .tool, .content = "b", .tool_call_id = 2 },
    };
    try validate(alloc, &answered, &.{}, .{});
    const continued = [_]Message{
        .{ .role = .user, .content = "go" },
        .{ .role = .assistant, .content = "thinking", .tool_calls = &calls },
        .{ .role = .tool, .content = "a", .tool_call_id = 1 },
        .{ .role = .tool, .content = "b", .tool_call_id = 2 },
        .{ .role = .user, .content = "next" },
    };
    try validate(alloc, &continued, &.{}, .{});
}

test "invalid tool history is rejected before rendering" {
    const alloc = std.testing.allocator;
    const call = ToolCall{ .id = 1, .name = "read_file", .arguments = "{}" };
    const two = [_]ToolCall{ call, .{ .id = 2, .name = "glob", .arguments = "{}" } };
    const assistant: Message = .{ .role = .assistant, .content = "", .tool_calls = &.{call} };
    const assistant2: Message = .{ .role = .assistant, .content = "", .tool_calls = &two };
    const result: Message = .{ .role = .tool, .content = "x", .tool_call_id = 1 };

    // Orphan result: nothing pending.
    try std.testing.expectError(error.InvalidConversation, validate(alloc, &.{ user_message, result }, &.{}, .{}));
    // Duplicate result: the group is already complete.
    try std.testing.expectError(error.InvalidConversation, validate(alloc, &.{ user_message, assistant, result, result }, &.{}, .{}));
    // Missing result: a user message arrives before the group completes.
    try std.testing.expectError(error.InvalidConversation, validate(alloc, &.{ user_message, assistant, user_message }, &.{}, .{}));
    // Missing result at the end (a conversation may not end on the assistant).
    try std.testing.expectError(error.InvalidConversation, validate(alloc, &.{ user_message, assistant }, &.{}, .{}));
    // Out of order: the second pending call is answered first.
    try std.testing.expectError(error.InvalidConversation, validate(alloc, &.{ user_message, assistant2, .{ .role = .tool, .content = "x", .tool_call_id = 2 } }, &.{}, .{}));
    // A tool result must carry the pending call's ID.
    try std.testing.expectError(error.InvalidConversation, validate(alloc, &.{ user_message, assistant, .{ .role = .tool, .content = "x" } }, &.{}, .{}));
    // Duplicate call IDs within one assistant message.
    const dup = [_]ToolCall{ call, .{ .id = 1, .name = "glob", .arguments = "{}" } };
    try std.testing.expectError(error.InvalidConversation, validate(alloc, &.{ user_message, .{ .role = .assistant, .content = "", .tool_calls = &dup } }, &.{}, .{}));
    // Only a `.tool` message carries a correlation ID.
    try std.testing.expectError(error.InvalidConversation, validate(alloc, &.{.{ .role = .user, .content = "x", .tool_call_id = 1 }}, &.{}, .{}));
    // Calls are only for the assistant.
    try std.testing.expectError(error.InvalidConversation, validate(alloc, &.{.{ .role = .user, .content = "x", .tool_calls = &.{call} }}, &.{}, .{}));
}

test "tool arguments and schemas must be JSON objects" {
    const alloc = std.testing.allocator;
    const bad_array = Message{ .role = .assistant, .content = "", .tool_calls = &.{.{ .id = 1, .name = "x", .arguments = "[]" }} };
    const bad_truncated = Message{ .role = .assistant, .content = "", .tool_calls = &.{.{ .id = 1, .name = "x", .arguments = "{" }} };
    try std.testing.expectError(error.InvalidConversation, validate(alloc, &.{ user_message, bad_array }, &.{}, .{}));
    try std.testing.expectError(error.InvalidConversation, validate(alloc, &.{ user_message, bad_truncated }, &.{}, .{}));
    try std.testing.expectError(error.InvalidConversation, validate(alloc, &.{user_message}, &.{.{ .name = "x", .description = "d", .parameters = "\"str\"" }}, .{}));
    try std.testing.expectError(error.InvalidConversation, validate(alloc, &.{user_message}, &.{.{ .name = "", .description = "d", .parameters = "{}" }}, .{}));
    const a = ToolDefinition{ .name = "dup", .description = "a", .parameters = "{}" };
    const b = ToolDefinition{ .name = "dup", .description = "b", .parameters = "{}" };
    try std.testing.expectError(error.InvalidConversation, validate(alloc, &.{user_message}, &.{ a, b }, .{}));
}

test "tool inputs honor the message, byte, and tool-count limits" {
    const alloc = std.testing.allocator;
    const tool = ToolDefinition{ .name = "x", .description = "d", .parameters = "{}" };
    try std.testing.expectError(error.LimitExceeded, validate(alloc, &.{user_message}, &.{ tool, tool }, .{ .tools = 1 }));
    try std.testing.expectError(error.LimitExceeded, validate(alloc, &.{user_message}, &.{tool}, .{ .input_bytes = 1 }));
    try std.testing.expectError(error.LimitExceeded, validate(alloc, &.{user_message}, &.{}, .{ .messages = 0 }));
    try std.testing.expectError(error.LimitExceeded, validate(alloc, &.{ user_message, .{ .role = .assistant, .content = "", .tool_calls = &.{read_call} } }, &.{}, .{ .input_bytes = 4 }));
}

test "leading-system counting and UTF-8 checking are shared" {
    try std.testing.expectEqual(0, leadingSystemCount(&.{}));
    try std.testing.expectEqual(2, leadingSystemCount(&.{ .{ .role = .system, .content = "a" }, .{ .role = .developer, .content = "b" }, user_message }));
    const alloc = std.testing.allocator;
    try std.testing.expectError(error.InvalidUtf8, validate(alloc, &.{.{ .role = .user, .content = "\xff" }}, &.{}, .{}));
    try std.testing.expectError(error.UnsupportedContent, validate(alloc, &.{.{ .role = .user, .content = "x", .reasoning_content = "y" }}, &.{}, .{}));
}

test {
    _ = stream;
    _ = qwen38;
    _ = gemma4;
}
