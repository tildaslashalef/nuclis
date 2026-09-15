//! `nuclis tokenize`: the prompt exactly as `generate` and `bench` would feed
//! it to the model, without building the model. The vocabulary and the chat
//! template live in the GGUF header, so the artifact is opened for its
//! directory only (the weights are mapped, never read). The report carries
//! the token IDs and, for each token, the byte offset where its text starts
//! in the rendered prompt: a benchmark script can cut a corpus at a token
//! boundary and verify a prompt's token count before any model run.
const std = @import("std");
const inference = @import("inference");
const engine = @import("engine.zig");
const generate = @import("generate.zig");
const style = @import("tui/style.zig");

pub const Report = struct {
    schema_version: u32 = 1,
    model_path: []const u8,
    /// `raw` (the text as given) or `chat` (one user turn rendered with the pinned profile).
    rendering: []const u8,
    /// Reasoning effort of the chat rendering; absent for raw prompts.
    think: ?[]const u8,
    /// Bytes of the rendered prompt (the template included for `chat`).
    bytes: usize,
    tokens: usize,
    ids: []const u32,
    /// `offsets[i]` is where token `i`'s decoded text starts in the rendered
    /// prompt; `offsets[tokens]` is the decoded length, equal to `bytes` when
    /// `round_trip` holds.
    offsets: []const usize,
    /// Whether decoding the IDs reproduces the rendered prompt byte for byte.
    /// False means an offset is not a cut point in the prompt text.
    round_trip: bool,

    pub fn render(self: Report, out: *std.Io.Writer, json: bool, sty: style.Style) !void {
        if (json) {
            try std.json.Stringify.value(self, .{ .emit_null_optional_fields = false }, out);
            return out.writeByte('\n');
        }
        const label = sty.on(.label);
        const number = sty.on(.number);
        const off = sty.off();
        try out.print("{s}Model:{s} {s}{s}{s}\n{s}Prompt:{s} {s}{d}{s} tokens, {s}{d}{s} bytes, {s} rendering", .{ label, off, sty.on(.code), self.model_path, off, label, off, number, self.tokens, off, number, self.bytes, off, self.rendering });
        if (self.think) |effort| try out.print(" (think {s})", .{effort});
        if (self.round_trip) try out.print(", round trip {s}ok{s}\n", .{ sty.on(.success), off }) else try out.print(", round trip {s}MISMATCH{s}\n", .{ sty.on(.error_text), off });
        try out.print("{s}ids:{s}", .{ label, off });
        for (self.ids) |id| try out.print(" {d}", .{id});
        try out.writeByte('\n');
    }
};

/// Byte offset of every token's decoded text plus the total, so a caller can
/// cut the decoded prompt at token `i` with `text[0..offsets[i]]`. Special
/// tokens count their marker text, as the encoder consumed it. Caller frees.
pub fn tokenOffsets(alloc: std.mem.Allocator, vocab: *const inference.vocabulary.Vocabulary, ids: []const u32) ![]usize {
    const offsets = try alloc.alloc(usize, ids.len + 1);
    errdefer alloc.free(offsets);
    var offset: usize = 0;
    for (ids, 0..) |_, i| {
        offsets[i] = offset;
        const piece = try inference.bpe.decode(alloc, vocab, ids[i..][0..1], true, .{});
        defer alloc.free(piece);
        offset += piece.len;
    }
    offsets[ids.len] = offset;
    return offsets;
}

pub fn run(alloc: std.mem.Allocator, io: std.Io, model_path: []const u8, effort: inference.profiles.Effort, options: generate.Options, json: bool, writer: *std.Io.Writer, sty: style.Style) !void {
    const user = try engine.readPrompt(alloc, io, options.prompt, options.prompt_file);
    defer alloc.free(user);
    var mapped = try inference.weights.Mapped.open(alloc, io, model_path);
    defer mapped.deinit(io);
    var vocab = try inference.vocabulary.load(alloc, mapped.document, mapped.mapping.memory[0..@intCast(mapped.document.directory_bytes)], .{});
    defer vocab.deinit();
    // The same two paths as `Engine.prompt`: the text unchanged, or one user
    // turn through the pinned renderer (refused when the template differs).
    const prompt = if (options.raw) try alloc.dupe(u8, user) else blk: {
        const profile = inference.profiles.forDocument(mapped.document) orelse return error.UnsupportedPromptTemplate;
        break :blk try profile.render(alloc, &.{.{ .role = .user, .content = user }}, &.{}, effort, .{});
    };
    defer alloc.free(prompt);
    var encoder = try inference.tokenizer.Encoder.init(alloc, &vocab);
    defer encoder.deinit();
    const ids = try encoder.encode(alloc, prompt, true, .{});
    defer alloc.free(ids);
    if (ids.len == 0) return error.EmptyPrompt;
    const offsets = try tokenOffsets(alloc, &vocab, ids);
    defer alloc.free(offsets);
    const decoded = try inference.bpe.decode(alloc, &vocab, ids, true, .{});
    defer alloc.free(decoded);
    const report: Report = .{
        .model_path = model_path,
        .rendering = if (options.raw) "raw" else "chat",
        .think = if (options.raw) null else @tagName(effort),
        .bytes = prompt.len,
        .tokens = ids.len,
        .ids = ids,
        .offsets = offsets,
        .round_trip = std.mem.eql(u8, decoded, prompt),
    };
    try report.render(writer, json, sty);
}

fn fixtureVocabulary() !inference.vocabulary.Vocabulary {
    // Token text is in the byte alphabet ('Ġ' is a space); the control marker is literal.
    const tokens = &[_]inference.vocabulary.Token{
        .{ .text = "a", .kind = .normal },
        .{ .text = "Ġb", .kind = .normal },
        .{ .text = "<x>", .kind = .control },
    };
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    errdefer arena.deinit();
    const alloc = arena.allocator();
    var ids: std.StringHashMapUnmanaged(u32) = .empty;
    for (tokens, 0..) |token, id| try ids.put(alloc, token.text, @intCast(id));
    return .{ .storage = arena, .tokens = tokens, .pre = "qwen35", .bos = null, .eos = null, .padding = null, .token_ids = ids, .merge_ranks = .empty };
}

test "offsets are the byte starts of each token's decoded text plus the total" {
    var vocab = try fixtureVocabulary();
    defer vocab.deinit();
    // "a" + "<x>" + " b" = "a<x> b": starts 0, 1, 4; total 6.
    const offsets = try tokenOffsets(std.testing.allocator, &vocab, &.{ 0, 2, 1 });
    defer std.testing.allocator.free(offsets);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 4, 6 }, offsets);
    const none = try tokenOffsets(std.testing.allocator, &vocab, &.{});
    defer std.testing.allocator.free(none);
    try std.testing.expectEqualSlices(usize, &.{0}, none);
    try std.testing.expectError(error.InvalidTokenId, tokenOffsets(std.testing.allocator, &vocab, &.{9}));
}

test "report renders text and JSON from the same value" {
    const report: Report = .{ .model_path = "m.gguf", .rendering = "raw", .think = null, .bytes = 6, .tokens = 3, .ids = &.{ 0, 2, 1 }, .offsets = &.{ 0, 1, 4, 6 }, .round_trip = true };
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try report.render(&out.writer, false, .none);
    try std.testing.expectEqualStrings("Model: m.gguf\nPrompt: 3 tokens, 6 bytes, raw rendering, round trip ok\nids: 0 2 1\n", out.written());
    out.clearRetainingCapacity();
    try report.render(&out.writer, true, .none);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, out.written(), .{});
    defer parsed.deinit();
    try std.testing.expect(parsed.value.object.get("think") == null);
    try std.testing.expectEqual(@as(i64, 3), parsed.value.object.get("tokens").?.integer);
    try std.testing.expectEqual(@as(usize, 4), parsed.value.object.get("offsets").?.array.items.len);
    const chat: Report = .{ .model_path = "m.gguf", .rendering = "chat", .think = "low", .bytes = 1, .tokens = 1, .ids = &.{0}, .offsets = &.{ 0, 1 }, .round_trip = false };
    out.clearRetainingCapacity();
    try chat.render(&out.writer, false, .none);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "chat rendering (think low), round trip MISMATCH") != null);
}
