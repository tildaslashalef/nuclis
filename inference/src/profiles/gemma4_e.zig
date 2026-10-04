//! Prompt profile for the Gemma 4 E-series template (the E4B's
//! `tokenizer.chat_template`): the pinned Gemma 4 profile (`gemma4.zig`)
//! with one rendered difference, read from the two templates' diff — with
//! thinking off, the generation prompt opens no empty thought channel.
//! Everything else (turns, tools, reasoning, sampling, markers) is Gemma 4's;
//! docs/reference/prompt-profile.md § Gemma 4 E-series.
const std = @import("std");
const gemma4 = @import("gemma4.zig");
const profiles = @import("root.zig");

/// Exact GGUF template this profile implements (18,810 bytes).
pub const template_sha256 = "241c50d86bdfe5e43307da87f559cd2416aacd67a8de46c15acc0105ef2200b7";
pub const template_aliases = [_][]const u8{};

pub const stop_tokens = gemma4.stop_tokens;
pub const bos_token = gemma4.bos_token;
pub const image_placeholder = gemma4.image_placeholder;
pub const reasoning = gemma4.reasoning;
pub const stream_markers = gemma4.stream_markers;
pub const efforts = gemma4.efforts;
pub const samplingDefaults = gemma4.samplingDefaults;
pub const prefix = gemma4.prefix;

pub fn render(alloc: std.mem.Allocator, messages: []const profiles.Message, tools: []const profiles.ToolDefinition, effort: profiles.Effort, limits: profiles.Limits) profiles.Error![]u8 {
    return gemma4.renderWith(alloc, messages, tools, effort, limits, .{ .empty_thought_when_off = false });
}

test "thinking off opens no thought channel; thinking on renders as Gemma 4" {
    const alloc = std.testing.allocator;
    const messages = [_]profiles.Message{.{ .role = .user, .content = "Hi" }};
    const off = try render(alloc, &messages, &.{}, .off, .{});
    defer alloc.free(off);
    try std.testing.expectEqualStrings("<bos><|turn>user\nHi<turn|>\n<|turn>model\n", off);
    const on = try render(alloc, &messages, &.{}, .medium, .{});
    defer alloc.free(on);
    const pinned = try gemma4.render(alloc, &messages, &.{}, .medium, .{});
    defer alloc.free(pinned);
    try std.testing.expectEqualStrings(pinned, on);
}

const Fixture = struct {
    template_sha256: []const u8,
    prompt_cases: []const struct { name: []const u8, effort: profiles.Effort, messages: []const profiles.Message, prompt: []const u8 },
};

test "text prompts match the E4B reference fixtures, `<bos>` prepended" {
    const alloc = std.testing.allocator;
    const fixture = try std.json.parseFromSlice(Fixture, alloc, @embedFile("fixtures/gemma4_e-text.json"), .{ .ignore_unknown_fields = true });
    defer fixture.deinit();
    try std.testing.expectEqualStrings(template_sha256, fixture.value.template_sha256);
    try std.testing.expectEqual(@as(usize, 14), fixture.value.prompt_cases.len);
    for (fixture.value.prompt_cases) |case| {
        const expected = try std.mem.concat(alloc, u8, &.{ "<bos>", case.prompt });
        defer alloc.free(expected);
        const prompt = try render(alloc, case.messages, &.{}, case.effort, .{});
        defer alloc.free(prompt);
        try std.testing.expectEqualStrings(expected, prompt);
    }
}
