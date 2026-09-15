//! Bounded prompt profile for the pinned Gemma 4 artifacts, implemented from
//! the chat template embedded in the GGUF and verified against prompts the
//! pinned llama.cpp server rendered from it. The template text is not in the
//! tree; `fixtures/gemma4-text.json` is its evidence. The behavior it encodes
//! — turns, the thinking switch, dropped reasoning, and the `<bos>` prefix —
//! is specified with fixture-backed clauses in docs/reference/prompt-profile.md.
//!
//! Text-only. Tool calling is deliberately unsupported: Gemma's grammar and
//! result handoff differ from Qwen's and need their own fixtures, so any tool
//! input returns `error.ToolsUnsupported` (docs/reference/tool-calling.md).
//! It owns no tokenizer or model equations.
const std = @import("std");
const sampling = @import("../sampling/root.zig");

/// Exact GGUF template this profile implements (`tokenizer.chat_template`,
/// 18,922 bytes ending in a newline); `profiles.forDocument` selects the
/// profile by this digest, never by architecture.
pub const template_sha256 = "845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b";

const profiles = @import("root.zig");
pub const Role = profiles.Role;
pub const Effort = profiles.Effort;
pub const Message = profiles.Message;
pub const Limits = profiles.Limits;
pub const Error = profiles.Error;
pub const SamplingOverrides = profiles.SamplingOverrides;

/// The tokens that end a model turn: `<turn|>` (106, the K-quant file's
/// `eos_token_id`) and `<eos>` (1, the QAT file's). Both files carry both.
pub const stop_tokens = [_][]const u8{ "<turn|>", "<eos>" };

/// How the model delimits its reasoning in generated text: a thought
/// channel opened by `<|channel>thought\n` and closed by `<channel|>`. Both
/// markers are user-defined tokens the decoder always renders as text.
pub const reasoning: profiles.Reasoning = .{ .open = "<|channel>thought\n", .close = "<channel|>" };
pub const stream_markers: profiles.StreamMarkers = .{ .open = "<|channel>", .close = "<channel|>", .open_suffix = "thought\n" };

/// Returns an owned UTF-8 prompt; caller frees it with the supplied allocator.
/// All message strings are borrowed only for this call. `profiles.validate`
/// enforces the shared conversation rules (completion-ready ending, role
/// order, bounds, tool history); a tool-free conversation still ends on the
/// user. Assistant reasoning is accepted separately and dropped (see the
/// module comment); assistant content carrying the channel markers is rejected
/// rather than stripped as the template would.
/// The bytes every rendering that starts with these system messages starts
/// with: `<bos>` and the system turn. Pinned as a byte prefix of `render`.
pub fn prefix(alloc: std.mem.Allocator, messages: []const Message, tools: []const profiles.ToolDefinition, effort: Effort, limits: Limits) Error![]u8 {
    if (tools.len != 0) return error.ToolsUnsupported;
    const prefix_count = profiles.leadingSystemCount(messages);
    const thinking = effort != .off;
    var builder: Builder = .{ .alloc = alloc, .limit = limits.output_bytes };
    errdefer builder.bytes.deinit(alloc);
    try builder.add("<bos>");
    if (thinking or prefix_count > 0) {
        try builder.add("<|turn>system\n");
        if (thinking) try builder.add("<|think|>\n");
        if (prefix_count > 0) try builder.add(trim(messages[0].content));
        try builder.add("<turn|>\n");
    }
    return builder.bytes.toOwnedSlice(alloc);
}

pub fn render(alloc: std.mem.Allocator, messages: []const Message, tools: []const profiles.ToolDefinition, effort: Effort, limits: Limits) Error![]u8 {
    try profiles.validate(alloc, messages, tools, limits);
    // The deliberate position recorded in the module comment: a structurally
    // valid tool input is rejected, not rendered in the wrong grammar.
    if (profiles.usesToolFeatures(messages, tools)) return error.ToolsUnsupported;
    for (messages) |message| {
        if (message.role == .assistant and
            (std.mem.indexOf(u8, message.content, reasoning.open[0.."<|channel>".len]) != null or
                std.mem.indexOf(u8, message.content, reasoning.close) != null)) return error.UnsupportedContent;
    }
    const prefix_count = profiles.leadingSystemCount(messages);

    const thinking = effort != .off;
    var builder: Builder = .{ .alloc = alloc, .limit = limits.output_bytes };
    errdefer builder.bytes.deinit(alloc);
    try builder.add("<bos>");
    // The system turn exists for thinking or for a leading system message;
    // only the first message is folded into it.
    if (thinking or prefix_count > 0) {
        try builder.add("<|turn>system\n");
        if (thinking) try builder.add("<|think|>\n");
        if (prefix_count > 0) try builder.add(trim(messages[0].content));
        try builder.add("<turn|>\n");
    }
    const rest = if (prefix_count > 0) messages[1..] else messages;
    var previous: ?Role = null;
    for (rest, 0..) |message, i| {
        const continued = message.role == .assistant and previous == .assistant;
        if (!continued) {
            try builder.add("<|turn>");
            try builder.add(roleName(message.role));
            try builder.add("\n");
        }
        try builder.add(trim(message.content));
        const continues = message.role == .assistant and i + 1 < rest.len and rest[i + 1].role == .assistant;
        if (!continues) try builder.add("<turn|>\n");
        previous = message.role;
    }
    try builder.add("<|turn>model\n");
    if (!thinking) {
        try builder.add(reasoning.open);
        try builder.add(reasoning.close);
    }
    return builder.bytes.toOwnedSlice(alloc);
}

/// The template's role names: the assistant is `model`, and the reference
/// renders `developer` as `system`.
fn roleName(role: Role) []const u8 {
    return switch (role) {
        .system, .developer => "system",
        .user => "user",
        .assistant => "model",
        // Unreachable: `render` rejects any tool history before it names turns.
        .tool => "tool",
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
}

test "prompt rendering cleans up every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationCase, .{});
}

test "the prefix is a byte prefix of every rendering that starts with its system message" {
    const alloc = std.testing.allocator;
    const system: Message = .{ .role = .system, .content = "You are terse." };
    const user: Message = .{ .role = .user, .content = "hello" };
    inline for (.{ .off, .low }) |effort| {
        for ([_][]const Message{ &.{ system, user }, &.{user} }) |messages| {
            const head = try prefix(alloc, messages[0 .. messages.len - 1], &.{}, effort, .{});
            defer alloc.free(head);
            const full = try render(alloc, messages, &.{}, effort, .{});
            defer alloc.free(full);
            try std.testing.expect(std.mem.startsWith(u8, full, head));
            try std.testing.expect(std.mem.startsWith(u8, full[head.len..], "<|turn>user\n"));
        }
    }
}
