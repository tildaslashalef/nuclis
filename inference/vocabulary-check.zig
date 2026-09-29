//! Explicit full-artifact check of the vocabulary, the tokenizer, and the
//! profile's captured prompts. No GPU, server, or tensor payload is needed.
//! A GGUF's template digest selects the profile, which selects the fixture
//! and the expected vocabulary facts; a directory is a Laya checkpoint, whose
//! `tokenizer/tokenizer.json` is checked against the oracle's fixtures.
const std = @import("std");
const inference = @import("inference");

const Expect = struct {
    tokens: usize,
    merges: usize,
    bos: ?u32,
    /// The EOS ids the profile's pinned files declare (Gemma's K-quant and
    /// QAT files differ).
    eos: []const u32,
    texts: []const []const u8,
    ids: []const u32,
    fixture: []const u8,
    /// Whether the saved single-piece inputs go through the byte-level
    /// `encodePiece` (GPT-2 vocabularies only).
    byte_level: bool,
};

const Selection = struct { name: []const u8, expect: Expect, profile: inference.profiles.Profile };

fn select(doc: inference.gguf.Document) ?Selection {
    const profile = inference.profiles.forDocument(doc) orelse return null;
    return .{ .name = @tagName(profile), .expect = expectations(profile), .profile = profile };
}

fn expectations(profile: inference.profiles.Profile) Expect {
    return switch (profile) {
        .muse_glimmer => .{
            .tokens = 202_048,
            .merges = 439_802,
            .bos = 200000,
            .eos = &.{200001},
            .texts = &.{ "<|begin_of_text|>", "<|end_of_text|>", "<|eom|>", "<|eot|>", "<|finetune_right_pad|>", "<|start|>", "<|message|>", "system", "user", "assistant", "tool" },
            .ids = &.{ 200000, 200001, 200007, 200008, 200018, 200022, 200023, 15651, 1556, 140680, 21188 },
            .fixture = @embedFile("src/profiles/fixtures/muse_glimmer-text.json"),
            .byte_level = true,
        },
        .qwen38 => .{
            .tokens = 248_320,
            .merges = 247_587,
            .bos = 248044,
            .eos = &.{248046},
            .texts = &.{ "<|im_start|>", "<|im_end|>", "<think>", "</think>", "user", "assistant" },
            .ids = &.{ 248045, 248046, 248068, 248069, 846, 74455 },
            .fixture = @embedFile("src/profiles/fixtures/qwen38-text.json"),
            .byte_level = true,
        },
        .gemma4, .gemma4_e => .{
            .tokens = 262_144,
            .merges = 514_906,
            .bos = 2,
            .eos = &.{ 106, 1 },
            .texts = &.{ "<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|think|>", "<eos>", "user", "model", "system", "thought", "<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>", "<|\"|>" },
            .ids = &.{ 105, 106, 100, 101, 98, 1, 2364, 4368, 9731, 45518, 46, 47, 48, 49, 50, 51, 52 },
            .fixture = if (profile == .gemma4_e) @embedFile("src/profiles/fixtures/gemma4_e-text.json") else @embedFile("src/profiles/fixtures/gemma4-text.json"),
            .byte_level = false,
        },
    };
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) return error.ExpectedModelPath;
    const stat = try std.Io.Dir.cwd().statFile(init.io, args[1], .{});
    if (stat.kind == .directory) return checkLaya(init.gpa, init.io, args[1]);
    var model = try inference.weights.Mapped.open(init.gpa, init.io, args[1]);
    defer model.deinit(init.io);
    const doc = model.document;
    const directory = model.mapping.memory[0..@intCast(doc.directory_bytes)];
    const selection = select(doc) orelse return error.UnsupportedPromptTemplate;
    const expect = selection.expect;
    var vocab = try inference.vocabulary.load(init.gpa, doc, directory, .{});
    defer vocab.deinit();
    if (vocab.tokens.len != expect.tokens or vocab.merge_ranks.count() != expect.merges or vocab.bos != expect.bos) return error.ArtifactMismatch;
    if (std.mem.indexOfScalar(u32, expect.eos, vocab.eos orelse return error.ArtifactMismatch) == null) return error.ArtifactMismatch;
    for (expect.texts, expect.ids) |text, id| {
        if (vocab.tokenId(text) != id) return error.TokenIdMismatch;
    }
    // The stop set the engine resolves at load must exist in this vocabulary.
    for (selection.profile.stopTokens()) |text| if (vocab.tokenId(text) == null) return error.MissingStopToken;
    std.debug.print("Vocabulary check passed ({s}): {d} tokens, {d} merges; selected fixture IDs match.\n", .{ selection.name, vocab.tokens.len, vocab.merge_ranks.count() });
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

const laya_tokens = @embedFile("src/models/fixtures/laya/tokens.json");
const laya_requests = @embedFile("src/models/fixtures/laya/requests.json");

/// The oracle's text set, and every text each fixture request tokenized: its
/// head, options (at most 48 tokens, maybe shrunk), and state (maybe cut at
/// the tail, or at the head for list states), found in the recorded sequence.
fn checkLaya(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    const path = try std.fs.path.join(gpa, &.{ dir, "tokenizer", "tokenizer.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 * 1024 * 1024));
    defer gpa.free(bytes);
    var tokenizer = try inference.hf_tokenizer.parse(gpa, bytes, .{});
    defer tokenizer.deinit();
    const vocab = &tokenizer.vocab;
    if (vocab.tokens.len != 50_368 or vocab.merge_ranks.count() != 50_009) return error.ArtifactMismatch;
    for ([_][]const u8{ "[CLS]", "[SEP]", "[PAD]", "[MASK]" }, [_]u32{ 50281, 50282, 50283, 50284 }) |text, id| {
        if (vocab.tokenId(text) != id) return error.TokenIdMismatch;
    }

    const Case = struct { text: []const u8, ids: []const u32 };
    const cases = try std.json.parseFromSlice(struct { token_cases: []const Case }, gpa, laya_tokens, .{ .ignore_unknown_fields = true });
    defer cases.deinit();
    var decoded_cases: usize = 0;
    for (cases.value.token_cases) |case| {
        const ids = try tokenizer.encode(gpa, case.text, .{});
        defer gpa.free(ids);
        if (!std.mem.eql(u32, ids, case.ids)) {
            std.debug.print("Mismatch for {f}: expected {any}, got {any}\n", .{ std.json.fmt(case.text, .{}), case.ids, ids });
            return error.TokenizationMismatch;
        }
        // Decoding restores text that NFC leaves alone and no marker strips.
        const normalized = try inference.hf_tokenizer.nfc.normalize(gpa, case.text);
        defer if (normalized) |n| gpa.free(n);
        if (normalized != null or std.mem.indexOf(u8, case.text, "[MASK]") != null) continue;
        const decoded = try inference.bpe.decode(gpa, vocab, ids, true, .{});
        defer gpa.free(decoded);
        if (!std.mem.eql(u8, decoded, case.text)) return error.DetokenizationMismatch;
        decoded_cases += 1;
    }

    const Request = struct { name: []const u8, ids: []const u32, markers: []const usize, head_text: []const u8, options: []const []const u8, state_text: []const u8, truncate_left: bool };
    const requests = try std.json.parseFromSlice(struct { requests: []const Request }, gpa, laya_requests, .{ .ignore_unknown_fields = true });
    defer requests.deinit();
    const sep = vocab.tokenId("[SEP]").?;
    for (requests.value.requests) |r| {
        const first = r.markers[0];
        try expectPrefix(gpa, &tokenizer, r.name, r.head_text, r.ids[1 .. first - 1]);
        if (r.ids[first - 1] != sep) return error.SequenceMismatch;
        const state_start = std.mem.indexOfScalarPos(u32, r.ids, r.markers[r.markers.len - 1], sep).? + 1;
        for (r.markers, r.options, 0..) |at, option, i| {
            const end = if (i + 1 < r.markers.len) r.markers[i + 1] else state_start - 1;
            const text = try std.mem.concat(gpa, u8, &.{ " ", option });
            defer gpa.free(text);
            if (end - at - 1 > 48) return error.SequenceMismatch;
            try expectPrefix(gpa, &tokenizer, r.name, text, r.ids[at + 1 .. end]);
        }
        const state = r.ids[state_start .. r.ids.len - 1];
        const all = try tokenizer.encode(gpa, r.state_text, .{});
        defer gpa.free(all);
        const kept = if (r.truncate_left) all[all.len - state.len ..] else all[0..state.len];
        if (state.len > all.len or !std.mem.eql(u32, kept, state)) {
            std.debug.print("{s}: state tokens differ\n", .{r.name});
            return error.TokenizationMismatch;
        }
    }
    std.debug.print("Laya tokenizer check passed: {d} tokens, {d} merges; {d} text cases ({d} decoded back), {d} request sequences.\n", .{ vocab.tokens.len, vocab.merge_ranks.count(), cases.value.token_cases.len, decoded_cases, requests.value.requests.len });
}

/// `text` must encode to ids that begin with `recorded` (budgets only cut).
fn expectPrefix(gpa: std.mem.Allocator, tokenizer: *const inference.hf_tokenizer.Tokenizer, name: []const u8, text: []const u8, recorded: []const u32) !void {
    const ids = try tokenizer.encode(gpa, text, .{});
    defer gpa.free(ids);
    if (recorded.len > ids.len or !std.mem.eql(u32, ids[0..recorded.len], recorded)) {
        std.debug.print("{s}: {f} encodes to {any}, sequence holds {any}\n", .{ name, std.json.fmt(text, .{}), ids, recorded });
        return error.TokenizationMismatch;
    }
}
