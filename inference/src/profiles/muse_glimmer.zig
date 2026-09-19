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
//! always opens its reasoning message). Tool calling (the ATEM protocol)
//! is a later unit: a tool input is `error.ToolsUnsupported` today.
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
pub const reasoning: profiles.Reasoning = .{ .open = "<|start|>assistant to=self<|message|>", .close = "<|eom|>" };
pub const stream_markers: profiles.StreamMarkers = .{
    .open = "<|start|>",
    .close = "<|eom|>",
    .channel = .{ .start = "<|start|>", .message = "<|message|>", .eom = "<|eom|>" },
};

/// The template's opening text: the encoder never adds BOS, so the profile
/// writes it (the reference server's `/apply-template` strips it).
const bos = "<|begin_of_text|>";
/// The system turn the template synthesizes when no system message exists,
/// without its date line.
const default_system = "You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.";
/// Every control marker the template spells. Content carrying one is
/// rejected rather than rendered, so structure cannot be smuggled past the
/// profile; the assistant's own past text loses them instead.
const markers = [_][]const u8{ "<|start|>", "<|message|>", "<|eom|>", "<|eot|>", "<|end_of_text|>", bos };

/// The bytes every rendering that starts with these system messages begins
/// with: `<|begin_of_text|>` and the system turns (the synthesized one when
/// there are none). Pinned as a byte prefix of `render`.
pub fn prefix(alloc: std.mem.Allocator, messages: []const Message, tools: []const profiles.ToolDefinition, effort: Effort, limits: Limits) Error![]u8 {
    if (tools.len != 0) return error.ToolsUnsupported;
    var builder: Builder = .{ .alloc = alloc, .limit = limits.output_bytes };
    errdefer builder.bytes.deinit(alloc);
    try renderSystem(&builder, messages[0..profiles.leadingSystemCount(messages)], effort);
    return builder.bytes.toOwnedSlice(alloc);
}

/// Returns an owned UTF-8 prompt; caller frees it with the supplied allocator.
/// All message strings are borrowed only for this call. `profiles.validate`
/// enforces the shared conversation rules. Nothing is trimmed: the template
/// writes content and reasoning verbatim, and the model's own reasoning
/// bytes end where `<|eom|>` begins.
pub fn render(alloc: std.mem.Allocator, messages: []const Message, tools: []const profiles.ToolDefinition, effort: Effort, limits: Limits) Error![]u8 {
    try profiles.validate(alloc, messages, tools, limits);
    if (profiles.usesToolFeatures(messages, tools)) return error.ToolsUnsupported;
    const prefix_count = profiles.leadingSystemCount(messages);
    var builder: Builder = .{ .alloc = alloc, .limit = limits.output_bytes };
    errdefer builder.bytes.deinit(alloc);
    try renderSystem(&builder, messages[0..prefix_count], effort);
    for (messages[prefix_count..]) |message| switch (message.role) {
        .system, .developer => unreachable, // `validate`: only leading
        .user => {
            if (hasMarker(message.content)) return error.UnsupportedContent;
            try builder.add("<|start|>user<|message|>");
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
            try builder.add("<|start|>assistant to=user<|message|>");
            try builder.add(own_content);
            try builder.add("<|eot|>");
        },
        .tool => unreachable, // `usesToolFeatures`
    };
    try builder.add("<|start|>assistant");
    return builder.bytes.toOwnedSlice(alloc);
}

/// `<|begin_of_text|>` and one system turn per leading system or developer
/// message (the reference renders both roles as `system`), each closed by
/// the strength and recipients lines; the synthesized turn when there are
/// none. Content is verbatim: the template does not trim.
fn renderSystem(builder: *Builder, leading: []const Message, effort: Effort) Error!void {
    try builder.add(bos);
    if (leading.len == 0) {
        try builder.add("<|start|>system<|message|>");
        try builder.add(default_system);
        try renderSystemMeta(builder, effort);
        return;
    }
    for (leading) |message| {
        if (hasMarker(message.content)) return error.UnsupportedContent;
        try builder.add("<|start|>system<|message|>");
        try builder.add(message.content);
        try renderSystemMeta(builder, effort);
    }
}

/// The lines the template appends to every system turn.
fn renderSystemMeta(builder: *Builder, effort: Effort) Error!void {
    try builder.add("\n\nReasoning strength: ");
    try builder.add(strengthName(effort));
    try builder.add(".\n\n# Valid recipients: \"self\", \"user\".<|eot|>");
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
    // Tools are a later unit: refused with the shared error.
    const tool: profiles.ToolDefinition = .{ .name = "read_file", .description = "d", .parameters = "{}" };
    try std.testing.expectError(error.ToolsUnsupported, render(alloc, &.{user}, &.{tool}, .low, .{}));
    try std.testing.expectError(error.ToolsUnsupported, prefix(alloc, &.{}, &.{tool}, .low, .{}));
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
    const head = try prefix(alloc, &.{.{ .role = .system, .content = "Be precise." }}, &.{}, .xhigh, .{});
    defer alloc.free(head);
}

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
            const head = try prefix(alloc, messages[0 .. messages.len - 1], &.{}, effort, .{});
            defer alloc.free(head);
            const full = try render(alloc, messages, &.{}, effort, .{});
            defer alloc.free(full);
            try std.testing.expect(std.mem.startsWith(u8, full, head));
            try std.testing.expect(std.mem.startsWith(u8, full[head.len..], "<|start|>user<|message|>"));
        }
    }
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
