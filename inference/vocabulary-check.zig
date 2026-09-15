//! Explicit full-artifact check of the vocabulary, the tokenizer, and the
//! profile's captured prompts. No GPU, server, or tensor payload is needed.
//! The artifact's template digest selects the profile and with it the
//! fixture and the expected vocabulary facts (both pinned files).
const std = @import("std");
const inference = @import("inference");

const Expect = struct {
    tokens: usize,
    merges: usize,
    bos: ?u32,
    eos: ?u32,
    texts: []const []const u8,
    ids: []const u32,
    fixture: []const u8,
    /// Whether the saved single-piece inputs go through the byte-level
    /// `encodePiece` (GPT-2 vocabularies only).
    byte_level: bool,
};

fn expectations(profile: inference.profiles.Profile) Expect {
    return switch (profile) {
        .qwen38 => .{
            .tokens = 248_320,
            .merges = 247_587,
            .bos = 248044,
            .eos = 248046,
            .texts = &.{ "<|im_start|>", "<|im_end|>", "<think>", "</think>", "user", "assistant" },
            .ids = &.{ 248045, 248046, 248068, 248069, 846, 74455 },
            .fixture = @embedFile("src/profiles/fixtures/qwen38-text.json"),
            .byte_level = true,
        },
        .gemma4 => .{
            .tokens = 262_144,
            .merges = 514_906,
            .bos = 2,
            .eos = 106,
            .texts = &.{ "<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|think|>", "<eos>", "user", "model", "system", "thought" },
            .ids = &.{ 105, 106, 100, 101, 98, 1, 2364, 4368, 9731, 45518 },
            .fixture = @embedFile("src/profiles/fixtures/gemma4-text.json"),
            .byte_level = false,
        },
    };
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) return error.ExpectedModelPath;
    var model = try inference.weights.Mapped.open(init.gpa, init.io, args[1]);
    defer model.deinit(init.io);
    const doc = model.document;
    const directory = model.mapping.memory[0..@intCast(doc.directory_bytes)];
    const profile = inference.profiles.forDocument(doc) orelse return error.UnsupportedPromptTemplate;
    const expect = expectations(profile);
    var vocab = try inference.vocabulary.load(init.gpa, doc, directory, .{});
    defer vocab.deinit();
    if (vocab.tokens.len != expect.tokens or vocab.merge_ranks.count() != expect.merges or
        vocab.bos != expect.bos or vocab.eos != expect.eos) return error.ArtifactMismatch;
    for (expect.texts, expect.ids) |text, id| {
        if (vocab.tokenId(text) != id) return error.TokenIdMismatch;
    }
    // The stop set the engine resolves at load must exist in this vocabulary.
    for (profile.stopTokens()) |text| if (vocab.tokenId(text) == null) return error.MissingStopToken;
    std.debug.print("Vocabulary check passed ({s}): {d} tokens, {d} merges; selected fixture IDs match.\n", .{ @tagName(profile), vocab.tokens.len, vocab.merge_ranks.count() });
    try checkFixtures(init.gpa, &vocab, expect);
}

fn checkFixtures(alloc: std.mem.Allocator, vocab: *const inference.vocabulary.Vocabulary, expect: Expect) !void {
    const Case = struct { text: []const u8, tokens: []const u32, parse_special: bool };
    const Prompt = struct { prompt: []const u8, tokens: []const u32 };
    const Fixture = struct { token_cases: []const Case, prompt_cases: []const Prompt };
    const fixtures = try std.json.parseFromSlice(Fixture, alloc, expect.fixture, .{ .ignore_unknown_fields = true });
    defer fixtures.deinit();
    var encoder = try inference.tokenizer.Encoder.init(alloc, vocab);
    defer encoder.deinit();
    const seen = try alloc.alloc(bool, vocab.tokens.len);
    defer alloc.free(seen);
    @memset(seen, false);
    var pieces: usize = 0;
    for (fixtures.value.token_cases) |case| {
        const ids = try encoder.encode(alloc, case.text, case.parse_special, .{});
        defer alloc.free(ids);
        if (!std.mem.eql(u32, ids, case.tokens)) {
            std.debug.print("Mismatch for {s}: expected {any}, got {any}\n", .{ case.text, case.tokens, ids });
            return error.TokenizationMismatch;
        }
        const decoded = try inference.bpe.decode(alloc, vocab, case.tokens, true, .{});
        defer alloc.free(decoded);
        if (!std.mem.eql(u8, decoded, case.text)) return error.DetokenizationMismatch;
        for (case.tokens) |id| seen[id] = true;
        // These saved inputs are single pre-tokenizer pieces (or empty), so
        // they can test the BPE core before Unicode splitting is implemented.
        if (expect.byte_level and (case.text.len == 0 or std.mem.eql(u8, case.text, "hello") or std.mem.eql(u8, case.text, " hello"))) {
            const piece_ids = try inference.bpe.encodePiece(alloc, vocab, case.text, .{});
            defer alloc.free(piece_ids);
            if (!std.mem.eql(u32, piece_ids, case.tokens)) return error.BpeMismatch;
            pieces += 1;
        }
    }
    for (fixtures.value.prompt_cases) |case| {
        const ids = try encoder.encode(alloc, case.prompt, true, .{});
        defer alloc.free(ids);
        if (!std.mem.eql(u32, ids, case.tokens)) return error.PromptTokenizationMismatch;
        const decoded = try inference.bpe.decode(alloc, vocab, case.tokens, true, .{});
        defer alloc.free(decoded);
        if (!std.mem.eql(u8, decoded, case.prompt)) return error.DetokenizationMismatch;
        for (case.tokens) |id| seen[id] = true;
    }
    var normal_pieces: usize = 0;
    for (seen, 0..) |present, index| {
        if (!present or vocab.tokens[index].kind != .normal or !expect.byte_level) continue;
        const id: u32 = @intCast(index);
        const raw = try inference.bpe.decode(alloc, vocab, &.{id}, true, .{});
        defer alloc.free(raw);
        const encoded = try inference.bpe.encodePiece(alloc, vocab, raw, .{});
        defer alloc.free(encoded);
        if (!std.mem.eql(u32, encoded, &.{id})) return error.TokenPieceMismatch;
        normal_pieces += 1;
    }
    std.debug.print("BPE check passed: {d} single-piece cases, {d} distinct normal token pieces; {d} captured token sequences decode exactly.\n", .{ pieces, normal_pieces, fixtures.value.token_cases.len + fixtures.value.prompt_cases.len });
    std.debug.print("Native encoding matches all {d} standalone and {d} full-prompt token sequences.\n", .{ fixtures.value.token_cases.len, fixtures.value.prompt_cases.len });
}
