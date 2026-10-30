//! Drama 3's text side: the tokenizer stored under `drama3.tokenizer.*`, the
//! input checks (Fish's `validate_drama3_text` plus the cue rules), the
//! `reference-bracket-v1` prompt rows, the long-text chunker (Fish's
//! `chunking.py`), and the chunk history. Pure logic over the GGUF
//! directory; no weights. docs/models/drama3.md § The prompt, § Chunking and
//! history.
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const vocabulary = @import("../tokenizer/vocabulary.zig");
const encode = @import("../tokenizer/encode.zig");
const nfc = @import("../tokenizer/nfc.zig");

const Allocator = std.mem.Allocator;

pub const codebooks = 10;
pub const codebook_size = 4096;
pub const max_speaker = 15;
pub const max_prompt_rows = 8192;

/// Codes 0..9 of one frame; code 0 is the main token's offset in the
/// semantic band.
pub const Codes = [codebooks]u16;

/// One prompt row. A text row carries its token id and zero codes; an audio
/// row's `main` is `semantic_start + codes[0]`.
pub const Row = struct { main: u32, codes: Codes = @splat(0) };

pub const InputError = error{
    EmptyText,
    ReservedText,
    InvalidSpeakerTag,
    UnknownSpecialToken,
    UnbalancedCue,
    NestedCue,
    UnclosedTag,
};

pub const Error = Allocator.Error || encode.Error || InputError || error{ PromptTooLong, InvalidCodes };

pub const ConfigError = vocabulary.Error || encode.Error || error{UnsupportedConfiguration};

/// Paired delivery tags; each must close before the text ends.
pub const paired_tags = [_][]const u8{ "whisper", "emphasis", "soft", "fast", "slow", "stress", "forceful" };

const template = "reference-bracket-v1";

/// The tokenizer and the template's literals, read once from the GGUF.
/// Owns its vocabulary and copies; the Document may be released after init.
pub const Text = struct {
    arena: std.heap.ArenaAllocator,
    vocab: *vocabulary.Vocabulary,
    encoder: encode.Encoder,
    eos: u32,
    semantic_start: u32,
    semantic_end: u32,
    system: []const u8,
    instruction: []const u8,
    reference_instruction: []const u8,
    reference_open: []const u8,
    reference_text: []const u8,
    reference_speech: []const u8,
    end: []const u8,
    user: []const u8,
    assistant: []const u8,
    reserved: []const []const u8,

    pub fn init(gpa: Allocator, doc: gguf.Document, directory: []const u8) ConfigError!Text {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        if (!std.mem.eql(u8, doc.string("drama3.prompt.template") orelse "", template)) return error.UnsupportedConfiguration;
        const eos = try unsigned(doc, "drama3.eos_token_id");
        const vocab = try a.create(vocabulary.Vocabulary);
        vocab.* = try vocabulary.loadKeys(gpa, doc, directory, .{
            .tokens = "drama3.tokenizer.tokens",
            .merges = "drama3.tokenizer.merges",
            .added_ids = "drama3.tokenizer.added_ids",
            .pre = "qwen35",
            .eos = eos,
            .padding = try unsigned(doc, "drama3.pad_token_id"),
        }, .{});
        errdefer vocab.deinit();
        var encoder = try encode.Encoder.init(gpa, vocab);
        errdefer encoder.deinit();
        // The arena is moved in last: everything above allocates from it.
        var self: Text = .{
            .arena = undefined,
            .vocab = vocab,
            .encoder = encoder,
            .eos = eos,
            .semantic_start = try unsigned(doc, "drama3.semantic_start_token_id"),
            .semantic_end = try unsigned(doc, "drama3.semantic_end_token_id"),
            .system = try literal(a, doc, "system"),
            .instruction = try literal(a, doc, "instruction"),
            .reference_instruction = try literal(a, doc, "reference_instruction"),
            .reference_open = try literal(a, doc, "reference_open"),
            .reference_text = try literal(a, doc, "reference_text"),
            .reference_speech = try literal(a, doc, "reference_speech"),
            .end = try literal(a, doc, "end"),
            .user = try literal(a, doc, "user"),
            .assistant = try literal(a, doc, "assistant"),
            .reserved = &.{},
        };
        const reserved = switch (doc.get("drama3.prompt.reserved") orelse return error.UnsupportedConfiguration) {
            .array => |r| r,
            else => return error.UnsupportedConfiguration,
        };
        if (reserved.element_type != .string or reserved.count == 0 or reserved.count > 64) return error.UnsupportedConfiguration;
        var reader = try gguf.ArrayReader.init(reserved, directory, .{});
        const list = try a.alloc([]const u8, @intCast(reserved.count));
        for (list) |*item| item.* = try a.dupe(u8, (try reader.nextString()) orelse return error.UnsupportedConfiguration);
        self.reserved = list;
        // Fish's own check (`build_drama3_prompt`): the band's ends and EOS are
        // the tokens the config names.
        if (self.semantic_end < self.semantic_start or self.semantic_end - self.semantic_start + 1 != codebook_size or
            vocab.tokenId("<|semantic:0|>") != self.semantic_start or
            vocab.tokenId("<|semantic:4095|>") != self.semantic_end or
            vocab.tokenId("<|im_end|>") != eos) return error.UnsupportedConfiguration;
        var buffer: [24]u8 = undefined;
        for (0..max_speaker + 1) |k| {
            const tag = std.fmt.bufPrint(&buffer, "<|speaker:{d}|>", .{k}) catch unreachable;
            if (vocab.tokenId(tag) == null) return error.UnsupportedConfiguration;
        }
        self.arena = arena;
        return self;
    }

    pub fn deinit(self: *Text) void {
        self.encoder.deinit();
        self.vocab.deinit();
        self.arena.deinit();
        self.* = undefined;
    }

    /// Owned ids of one literal. NFC first, as Qwen3.5's tokenizer normalizes;
    /// added tokens (`<|speaker:N|>`, `<|im_end|>`, ...) match in the text.
    pub fn encodeText(self: *const Text, gpa: Allocator, text: []const u8) (encode.Error || nfc.Error)![]u32 {
        const normalized = try nfc.normalize(gpa, text);
        defer if (normalized) |n| gpa.free(n);
        return self.encoder.encode(gpa, normalized orelse text, false, .{});
    }

    pub fn isSemantic(self: *const Text, id: u32) bool {
        return id >= self.semantic_start and id <= self.semantic_end;
    }

    /// The prompt rows for `text` after `references` and `history`, ending in
    /// the open assistant voice turn (Fish's `build_drama3_prompt`). Owned by
    /// the caller. Every literal is tokenized on its own.
    pub fn prompt(self: *const Text, gpa: Allocator, references: []const Reference, history: []const Turn, text: []const u8) Error![]Row {
        try self.validate(text);
        var rows: std.ArrayList(Row) = .empty;
        errdefer rows.deinit(gpa);
        var scratch: std.ArrayList(u8) = .empty;
        defer scratch.deinit(gpa);
        try self.literalRows(gpa, &rows, self.system);
        try self.literalRows(gpa, &rows, if (references.len == 0) self.instruction else self.reference_instruction);
        for (references, 1..) |reference, index| {
            try self.validate(reference.transcript);
            // The tag must follow "Text: " directly: leading white space moves
            // behind it.
            scratch.clearRetainingCapacity();
            try scratch.print(gpa, "{s}{d}{s}", .{ self.reference_open, index, self.reference_text });
            const body = trimStartWhite(reference.transcript);
            const leading = reference.transcript[0 .. reference.transcript.len - body.len];
            if (speakerTagAt(body, 0)) |tag| {
                try scratch.print(gpa, "{s}{s}{s}", .{ body[0..tag.len], leading, body[tag.len..] });
            } else {
                try scratch.print(gpa, "<|speaker:0|> {s}", .{reference.transcript});
            }
            try scratch.appendSlice(gpa, self.reference_speech);
            try self.literalRows(gpa, &rows, scratch.items);
            try self.audioRows(gpa, &rows, reference.codes);
        }
        try self.literalRows(gpa, &rows, self.end);
        for (history) |turn| {
            try self.validate(turn.text);
            try self.userRows(gpa, &rows, &scratch, turn.text);
            try self.literalRows(gpa, &rows, self.assistant);
            try self.audioRows(gpa, &rows, turn.codes);
            try self.literalRows(gpa, &rows, self.end);
        }
        try self.userRows(gpa, &rows, &scratch, text);
        try self.literalRows(gpa, &rows, self.assistant);
        if (rows.items.len > max_prompt_rows) return error.PromptTooLong;
        return rows.toOwnedSlice(gpa);
    }

    fn userRows(self: *const Text, gpa: Allocator, rows: *std.ArrayList(Row), scratch: *std.ArrayList(u8), text: []const u8) Error!void {
        try self.literalRows(gpa, rows, self.user);
        scratch.clearRetainingCapacity();
        if (speakerTagAt(trimStartWhite(text), 0) == null) try scratch.appendSlice(gpa, "<|speaker:0|> ");
        try scratch.appendSlice(gpa, text);
        try self.literalRows(gpa, rows, scratch.items);
        try self.literalRows(gpa, rows, self.end);
    }

    fn literalRows(self: *const Text, gpa: Allocator, rows: *std.ArrayList(Row), text: []const u8) Error!void {
        const ids = try self.encodeText(gpa, text);
        defer gpa.free(ids);
        if (rows.items.len + ids.len > max_prompt_rows) return error.PromptTooLong;
        try rows.ensureUnusedCapacity(gpa, ids.len);
        for (ids) |id| rows.appendAssumeCapacity(.{ .main = id });
    }

    fn audioRows(self: *const Text, gpa: Allocator, rows: *std.ArrayList(Row), codes: []const Codes) Error!void {
        if (codes.len == 0) return error.InvalidCodes;
        if (rows.items.len + codes.len > max_prompt_rows) return error.PromptTooLong;
        try rows.ensureUnusedCapacity(gpa, codes.len);
        for (codes) |frame| {
            for (frame) |c| if (c >= codebook_size) return error.InvalidCodes;
            rows.appendAssumeCapacity(.{ .main = self.semantic_start + frame[0], .codes = frame });
        }
    }

    /// Fish's `validate_drama3_text` (non-empty, no structural token, speaker
    /// tags `<|speaker:0|>`..`<|speaker:15|>` only) and the cue rules: no
    /// other `<|...|>` token, `[cues]` closed and not nested, paired tags
    /// (`paired_tags`) closed in order.
    pub fn validate(self: *const Text, text: []const u8) InputError!void {
        if (trimStartWhite(text).len == 0) return error.EmptyText;
        for (self.reserved) |token| if (std.mem.indexOf(u8, text, token) != null) return error.ReservedText;
        try validateCues(text);
    }
};

pub const Reference = struct { transcript: []const u8, codes: []const Codes };
pub const Turn = struct { text: []const u8, codes: []const Codes };

fn unsigned(doc: gguf.Document, key: []const u8) error{UnsupportedConfiguration}!u32 {
    return switch (doc.get(key) orelse return error.UnsupportedConfiguration) {
        .unsigned => |n| std.math.cast(u32, n) orelse error.UnsupportedConfiguration,
        else => error.UnsupportedConfiguration,
    };
}

fn literal(a: Allocator, doc: gguf.Document, comptime name: []const u8) (Allocator.Error || error{UnsupportedConfiguration})![]const u8 {
    const value = doc.string("drama3.prompt." ++ name) orelse return error.UnsupportedConfiguration;
    if (value.len == 0 or value.len > 256) return error.UnsupportedConfiguration;
    return a.dupe(u8, value);
}

/// The speaker tag `<|speaker:N|>` starting at `at`, canonical digits only.
fn speakerTagAt(text: []const u8, at: usize) ?struct { len: usize, speaker: u8 } {
    const open = "<|speaker:";
    if (!std.mem.startsWith(u8, text[at..], open)) return null;
    var i = at + open.len;
    const digits = i;
    while (i < text.len and std.ascii.isDigit(text[i])) i += 1;
    if (i == digits or i - digits > 2 or !std.mem.startsWith(u8, text[i..], "|>")) return null;
    if (text[digits] == '0' and i - digits > 1) return null;
    const n = std.fmt.parseInt(u8, text[digits..i], 10) catch return null;
    if (n > max_speaker) return null;
    return .{ .len = i + 2 - at, .speaker = n };
}

fn validateCues(text: []const u8) InputError!void {
    var open_bracket = false;
    var tags: [8]u8 = undefined;
    var depth: usize = 0;
    var i: usize = 0;
    while (i < text.len) {
        if (std.mem.startsWith(u8, text[i..], "<|")) {
            if (std.mem.startsWith(u8, text[i..], "<|speaker")) {
                const tag = speakerTagAt(text, i) orelse return error.InvalidSpeakerTag;
                i += tag.len;
                continue;
            }
            if (std.mem.indexOfPos(u8, text, i + 2, "|>") != null) return error.UnknownSpecialToken;
        }
        switch (text[i]) {
            '[' => {
                if (open_bracket) return error.NestedCue;
                open_bracket = true;
            },
            ']' => {
                if (!open_bracket) return error.UnbalancedCue;
                open_bracket = false;
            },
            '<' => if (pairedTag(text[i..])) |tag| {
                if (tag.closing) {
                    if (depth == 0 or tags[depth - 1] != tag.index) return error.UnclosedTag;
                    depth -= 1;
                } else {
                    if (depth == tags.len) return error.NestedCue;
                    tags[depth] = tag.index;
                    depth += 1;
                }
                i += tag.len;
                continue;
            },
            else => {},
        }
        i += 1;
    }
    if (open_bracket) return error.UnbalancedCue;
    if (depth != 0) return error.UnclosedTag;
}

fn pairedTag(text: []const u8) ?struct { index: u8, closing: bool, len: usize } {
    const closing = text.len > 1 and text[1] == '/';
    const name_at: usize = if (closing) 2 else 1;
    for (paired_tags, 0..) |name, index| {
        const rest = text[@min(name_at, text.len)..];
        if (std.mem.startsWith(u8, rest, name) and rest.len > name.len and rest[name.len] == '>')
            return .{ .index = @intCast(index), .closing = closing, .len = name_at + name.len + 1 };
    }
    return null;
}

/// The Unicode White_Space set Fish strips with (`UNICODE_WHITE_SPACE`).
fn isWhite(c: u21) bool {
    return switch (c) {
        0x9...0xD, 0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => true,
        else => false,
    };
}

fn trimStartWhite(text: []const u8) []const u8 {
    var it: std.unicode.Utf8Iterator = .{ .bytes = text, .i = 0 };
    var at: usize = 0;
    while (it.nextCodepoint()) |c| {
        if (!isWhite(c)) break;
        at = it.i;
    }
    return text[at..];
}

fn trimWhite(text: []const u8) []const u8 {
    const start = trimStartWhite(text);
    var end = start.len;
    while (end > 0) {
        var back = end - 1;
        while (back > 0 and start[back] & 0xC0 == 0x80) back -= 1;
        const c = std.unicode.utf8Decode(start[back..end]) catch break;
        if (!isWhite(c)) break;
        end = back;
    }
    return start[0..end];
}

// ---------------------------------------------------------------- chunking

pub const Chunking = struct {
    /// A threshold, not a cap: a protected span longer than this stays whole.
    bytes: usize = 300,
    speakers: usize = 3,
    /// Start a new chunk after every sentence end (nuclis's default: it cuts
    /// the model's stalls, docs/models/drama3.md § Stalls), or pack sentences
    /// up to `bytes` as Fish does.
    pack: bool = false,
};

const Piece = struct { text: []const u8, protected: bool };

/// The chunks of `text`, in order (Fish's `split_drama3_text`). Each chunk is
/// owned, as is the list.
pub fn split(gpa: Allocator, text: []const u8, options: Chunking) Allocator.Error![][]u8 {
    var out: std.ArrayList([]u8) = .empty;
    errdefer {
        for (out.items) |c| gpa.free(c);
        out.deinit(gpa);
    }
    var tagged = false;
    {
        var i: usize = 0;
        while (std.mem.indexOfPos(u8, text, i, "<|speaker:")) |at| : (i = at + 1) {
            if (speakerTagAt(text, at) != null) {
                tagged = true;
                break;
            }
        }
    }
    if (!tagged) {
        try splitByLength(gpa, text, options, &out);
        return out.toOwnedSlice(gpa);
    }
    // Speaker segments: text before the first tag is speaker 0.
    const Segment = struct { speaker: u8, text: []u8 };
    var segments: std.ArrayList(Segment) = .empty;
    defer {
        for (segments.items) |s| gpa.free(s.text);
        segments.deinit(gpa);
    }
    var speaker: u8 = 0;
    var body_start: usize = 0;
    var i: usize = 0;
    while (i <= text.len) {
        const tag = if (i < text.len) speakerTagAt(text, i) else null;
        if (i == text.len or tag != null) {
            const body = text[body_start..i];
            if (trimWhite(body).len != 0) {
                var pieces: std.ArrayList([]u8) = .empty;
                defer {
                    for (pieces.items) |p| gpa.free(p);
                    pieces.deinit(gpa);
                }
                try splitByLength(gpa, body, options, &pieces);
                for (pieces.items) |piece| {
                    const s = try std.fmt.allocPrint(gpa, "<|speaker:{d}|> {s}", .{ speaker, piece });
                    errdefer gpa.free(s);
                    try segments.append(gpa, .{ .speaker = speaker, .text = s });
                }
            }
            if (tag) |t| {
                speaker = t.speaker;
                i += t.len;
                body_start = i;
                continue;
            }
            break;
        }
        i += 1;
    }
    if (!options.pack) {
        for (segments.items) |*s| {
            try out.append(gpa, s.text);
            s.text = &.{};
        }
        return out.toOwnedSlice(gpa);
    }
    var batch: std.ArrayList(u8) = .empty;
    defer batch.deinit(gpa);
    var speakers: [max_speaker + 1]bool = @splat(false);
    var count: usize = 0;
    for (segments.items) |s| {
        const new_speaker = !speakers[s.speaker];
        if ((new_speaker and count >= options.speakers) or (batch.items.len != 0 and batch.items.len + s.text.len > options.bytes)) {
            try out.append(gpa, try batch.toOwnedSlice(gpa));
            speakers = @splat(false);
            count = 0;
        }
        if (batch.items.len != 0) try batch.append(gpa, '\n');
        try batch.appendSlice(gpa, s.text);
        if (!speakers[s.speaker]) {
            speakers[s.speaker] = true;
            count += 1;
        }
    }
    if (batch.items.len != 0) try out.append(gpa, try batch.toOwnedSlice(gpa));
    return out.toOwnedSlice(gpa);
}

// Fish's sets plus the Arabic-script question mark, semicolon, full stop and
// comma, which Fish's lack (a stall source, docs/models/drama3.md § Stalls).
const sentence_ends = ".!?;。！？；…\n؟؛۔";
const clause_marks = ",:，：、،";

/// Fish's `split_by_length`, appending owned chunks to `out`.
fn splitByLength(gpa: Allocator, text: []const u8, options: Chunking, out: *std.ArrayList([]u8)) Allocator.Error!void {
    if (trimWhite(text).len == 0) return;
    if (options.pack and text.len <= options.bytes) return out.append(gpa, try gpa.dupe(u8, text));
    var pieces: std.ArrayList(Piece) = .empty;
    defer pieces.deinit(gpa);
    try protectedPieces(gpa, text, &pieces);
    try breakAtMarks(gpa, &pieces, 0, sentence_ends);
    try breakAtMarks(gpa, &pieces, options.bytes, clause_marks);
    try breakAtMarks(gpa, &pieces, options.bytes, " ");
    try breakByLength(gpa, &pieces, options.bytes);

    var current: std.ArrayList(u8) = .empty;
    defer current.deinit(gpa);
    for (pieces.items, 0..) |piece, index| {
        const full = current.items.len != 0 and current.items.len + piece.text.len > options.bytes;
        const sentence = !options.pack and current.items.len != 0 and endsSentence(current.items) and speaksUntilSentenceEnd(pieces.items[index..]);
        if (full or sentence) try flush(gpa, &current, out);
        try current.appendSlice(gpa, piece.text);
    }
    try flush(gpa, &current, out);
}

fn flush(gpa: Allocator, current: *std.ArrayList(u8), out: *std.ArrayList([]u8)) Allocator.Error!void {
    const chunk = trimWhite(current.items);
    if (chunk.len != 0) try out.append(gpa, try gpa.dupe(u8, chunk));
    current.clearRetainingCapacity();
}

fn endsSentence(text: []const u8) bool {
    const t = trimWhite(text);
    if (t.len == 0) return false;
    var back = t.len - 1;
    while (back > 0 and t[back] & 0xC0 == 0x80) back -= 1;
    return std.mem.indexOf(u8, sentence_ends, t[back..]) != null;
}

/// Whether the pieces up to the next sentence end hold words to speak: a
/// trailing cue or stray punctuation stays with the sentence before it.
fn speaksUntilSentenceEnd(pieces: []const Piece) bool {
    for (pieces) |piece| {
        if (!piece.protected) {
            var it: std.unicode.Utf8Iterator = .{ .bytes = piece.text, .i = 0 };
            while (it.nextCodepoint()) |c| if (isWordChar(c)) return true;
        }
        if (endsSentence(piece.text)) return false;
    }
    return false;
}

fn isWordChar(c: u21) bool {
    if (c < 0x80) return std.ascii.isAlphanumeric(@intCast(c));
    return !isWhite(c) and std.mem.indexOfScalar(u21, &[_]u21{ '。', '！', '？', '；', '…', '，', '：', '、', '«', '»', '“', '”', '‘', '’', '–', '—', '؟', '،', '؛' }, c) == null;
}

/// Splits `text` into free and protected pieces: `<|...|>`, a paired tag
/// `<name ...>...</name>`, and `[...]` (Fish's `PROTECTED_SPAN`).
fn protectedPieces(gpa: Allocator, text: []const u8, pieces: *std.ArrayList(Piece)) Allocator.Error!void {
    var last: usize = 0;
    var i: usize = 0;
    while (i < text.len) {
        if (protectedAt(text, i)) |end| {
            if (i > last) try pieces.append(gpa, .{ .text = text[last..i], .protected = false });
            try pieces.append(gpa, .{ .text = text[i..end], .protected = true });
            i = end;
            last = end;
        } else i += 1;
    }
    if (last < text.len) try pieces.append(gpa, .{ .text = text[last..], .protected = false });
}

fn protectedAt(text: []const u8, i: usize) ?usize {
    switch (text[i]) {
        '<' => {
            if (i + 1 < text.len and text[i + 1] == '|') {
                var j = i + 2;
                while (j < text.len and text[j] != '|' and text[j] != '>') j += 1;
                if (j + 1 < text.len and text[j] == '|' and text[j + 1] == '>') return j + 2;
                return null;
            }
            if (i + 1 >= text.len or !std.ascii.isAlphabetic(text[i + 1])) return null;
            var j = i + 1;
            while (j < text.len and (std.ascii.isAlphanumeric(text[j]) or text[j] == '_' or text[j] == '-')) j += 1;
            const name = text[i + 1 .. j];
            const close = std.mem.indexOfScalarPos(u8, text, j, '>') orelse return null;
            var k = close + 1;
            while (std.mem.indexOfPos(u8, text, k, "</")) |at| : (k = at + 2) {
                if (std.mem.startsWith(u8, text[at + 2 ..], name) and at + 2 + name.len < text.len and text[at + 2 + name.len] == '>')
                    return at + 3 + name.len;
            }
            return null;
        },
        '[' => {
            var j = i + 1;
            while (j < text.len and text[j] != '[' and text[j] != ']') j += 1;
            return if (j < text.len and text[j] == ']') j + 1 else null;
        },
        else => return null,
    }
}

/// Splits free pieces longer than `length` bytes after each mark, keeping it.
fn breakAtMarks(gpa: Allocator, pieces: *std.ArrayList(Piece), length: usize, marks: []const u8) Allocator.Error!void {
    var out: std.ArrayList(Piece) = .empty;
    errdefer out.deinit(gpa);
    for (pieces.items) |piece| {
        if (piece.protected or piece.text.len <= length) {
            try out.append(gpa, piece);
            continue;
        }
        const t = piece.text;
        var start: usize = 0;
        var it: std.unicode.Utf8Iterator = .{ .bytes = t, .i = 0 };
        var prev: ?u21 = null;
        while (true) {
            const at = it.i;
            const c = it.nextCodepoint() orelse break;
            const next_at = it.i;
            var lookahead = it;
            const next = lookahead.nextCodepoint();
            if (isMark(t[at..next_at], marks) and !decimalPoint(c, prev, next, at, next_at, t.len)) {
                try out.append(gpa, .{ .text = t[start..next_at], .protected = false });
                start = next_at;
            }
            prev = c;
        }
        if (start < t.len) try out.append(gpa, .{ .text = t[start..], .protected = false });
    }
    pieces.deinit(gpa);
    pieces.* = out;
}

fn isMark(char: []const u8, marks: []const u8) bool {
    var it: std.unicode.Utf8Iterator = .{ .bytes = marks, .i = 0 };
    while (it.nextCodepointSlice()) |m| if (std.mem.eql(u8, m, char)) return true;
    return false;
}

/// Keeps decimals such as 3.14 in one piece.
fn decimalPoint(c: u21, prev: ?u21, next: ?u21, at: usize, next_at: usize, len: usize) bool {
    if (c != '.' or at == 0 or next_at >= len) return false;
    return isDigit(prev orelse return false) and isDigit(next orelse return false);
}

fn isDigit(c: u21) bool {
    return switch (c) {
        '0'...'9', 0x0660...0x0669, 0x06F0...0x06F9, 0xFF10...0xFF19 => true,
        else => false,
    };
}

/// Splits the free pieces still longer than `length` on character boundaries.
fn breakByLength(gpa: Allocator, pieces: *std.ArrayList(Piece), length: usize) Allocator.Error!void {
    var out: std.ArrayList(Piece) = .empty;
    errdefer out.deinit(gpa);
    for (pieces.items) |piece| {
        if (piece.protected or piece.text.len <= length) {
            try out.append(gpa, piece);
            continue;
        }
        const t = piece.text;
        var start: usize = 0;
        var it: std.unicode.Utf8Iterator = .{ .bytes = t, .i = 0 };
        while (true) {
            const at = it.i;
            _ = it.nextCodepointSlice() orelse break;
            if (at > start and it.i - start > length) {
                try out.append(gpa, .{ .text = t[start..at], .protected = false });
                start = at;
            }
        }
        try out.append(gpa, .{ .text = t[start..], .protected = false });
    }
    pieces.deinit(gpa);
    pieces.* = out;
}

// ---------------------------------------------------------------- history

/// Context for the next chunk (Fish's `Drama3History`): the first chunk
/// stays pinned, plus the latest `recent` chunks. Owns copies of each turn.
pub const History = struct {
    gpa: Allocator,
    turns: std.ArrayList(Owned) = .empty,
    recent: usize,

    const Owned = struct { text: []u8, codes: []Codes };

    pub fn init(gpa: Allocator, recent: usize) History {
        return .{ .gpa = gpa, .recent = recent };
    }

    pub fn deinit(self: *History) void {
        for (self.turns.items) |t| self.free(t);
        self.turns.deinit(self.gpa);
        self.* = undefined;
    }

    fn free(self: *History, t: Owned) void {
        self.gpa.free(t.text);
        self.gpa.free(t.codes);
    }

    pub fn push(self: *History, text: []const u8, codes: []const Codes) Allocator.Error!void {
        const owned: Owned = .{ .text = try self.gpa.dupe(u8, text), .codes = try self.gpa.dupe(Codes, codes) };
        errdefer self.free(owned);
        if (self.turns.items.len == 1 + self.recent) {
            if (self.recent == 0) return self.free(owned);
            self.free(self.turns.orderedRemove(1));
        }
        try self.turns.append(self.gpa, owned);
    }

    /// Candidate `index` of the histories to try, longest first: drop the
    /// oldest recent turns, then the pinned one, ending with none. Writes into
    /// `buffer` (at least `1 + recent` long); null past the last candidate.
    pub fn candidate(self: *const History, index: usize, buffer: []Turn) ?[]const Turn {
        const n = self.turns.items.len;
        if (n == 0) return if (index == 0) buffer[0..0] else null;
        const recent = n - 1;
        if (index <= recent) {
            buffer[0] = turn(self.turns.items[0]);
            var k: usize = 1;
            for (self.turns.items[1 + index ..]) |t| {
                buffer[k] = turn(t);
                k += 1;
            }
            return buffer[0..k];
        }
        return if (index == recent + 1) buffer[0..0] else null;
    }

    fn turn(t: Owned) Turn {
        return .{ .text = t.text, .codes = t.codes };
    }
};

// ---------------------------------------------------------------- tests

const testing = std.testing;

fn expectChunks(text: []const u8, options: Chunking, expected: []const []const u8) !void {
    const chunks = try split(testing.allocator, text, options);
    defer {
        for (chunks) |c| testing.allocator.free(c);
        testing.allocator.free(chunks);
    }
    try testing.expectEqual(expected.len, chunks.len);
    for (expected, chunks) |e, c| try testing.expectEqualStrings(e, c);
}

test "chunks split at every sentence end, keeping cues with their words" {
    try expectChunks("إزيك يا باشا؟ عامل إيه؟", .{}, &.{ "إزيك يا باشا؟", "عامل إيه؟" });
    try expectChunks("Wait... [sighs] Fine. It costs 3.14 dollars! [laughs]", .{}, &.{ "Wait...", "[sighs] Fine.", "It costs 3.14 dollars! [laughs]" });
    try expectChunks("طب استنى بس.", .{}, &.{"طب استنى بس."});
    try expectChunks("   \n ", .{}, &.{});
}

test "packed chunks follow Fish: short text whole, long text packed to the threshold" {
    try expectChunks("Hello there. How are you?", .{ .pack = true }, &.{"Hello there. How are you?"});
    try expectChunks("One. Two. Three.", .{ .pack = true, .bytes = 9 }, &.{ "One. Two.", "Three." });
    // A protected span is never split, even past the threshold.
    try expectChunks("[a very long cue that goes on] x", .{ .pack = true, .bytes = 8 }, &.{ "[a very long cue that goes on]", "x" });
}

test "speaker segments keep their tags; packing regroups up to three speakers" {
    const text = "Hi. <|speaker:1|> Hey! <|speaker:0|> Bye.";
    try expectChunks(text, .{}, &.{ "<|speaker:0|> Hi.", "<|speaker:1|> Hey!", "<|speaker:0|> Bye." });
    try expectChunks(text, .{ .pack = true }, &.{"<|speaker:0|> Hi. \n<|speaker:1|>  Hey! \n<|speaker:0|>  Bye."});
    try expectChunks("<|speaker:0|>a <|speaker:1|>b <|speaker:2|>c <|speaker:3|>d", .{ .pack = true }, &.{ "<|speaker:0|> a \n<|speaker:1|> b \n<|speaker:2|> c ", "<|speaker:3|> d" });
}

test "cue rules" {
    try validateCues("[sighs] <whisper>quiet</whisper> <|speaker:15|> ok");
    try testing.expectError(error.InvalidSpeakerTag, validateCues("<|speaker:16|> x"));
    try testing.expectError(error.InvalidSpeakerTag, validateCues("<|speaker:01|> x"));
    try testing.expectError(error.UnknownSpecialToken, validateCues("<|endoftext|>"));
    try testing.expectError(error.NestedCue, validateCues("[a [b]]"));
    try testing.expectError(error.UnbalancedCue, validateCues("a] b"));
    try testing.expectError(error.UnbalancedCue, validateCues("[open"));
    try testing.expectError(error.UnclosedTag, validateCues("<fast>go"));
    try testing.expectError(error.UnclosedTag, validateCues("<fast><slow>go</fast></slow>"));
    // An unknown angle bracket is text.
    try validateCues("a <3 b <unknown> c");
}

test "history keeps the first turn and the latest one, and offers shorter candidates" {
    var history: History = .init(testing.allocator, 1);
    defer history.deinit();
    var buffer: [2]Turn = undefined;
    try testing.expectEqual(@as(usize, 0), history.candidate(0, &buffer).?.len);
    try testing.expect(history.candidate(1, &buffer) == null);
    const codes = [_]Codes{@splat(1)};
    try history.push("one", &codes);
    try history.push("two", &codes);
    try history.push("three", &codes);
    const longest = history.candidate(0, &buffer).?;
    try testing.expectEqual(@as(usize, 2), longest.len);
    try testing.expectEqualStrings("one", longest[0].text);
    try testing.expectEqualStrings("three", longest[1].text);
    try testing.expectEqualStrings("one", history.candidate(1, &buffer).?[0].text);
    try testing.expectEqual(@as(usize, 0), history.candidate(2, &buffer).?.len);
    try testing.expect(history.candidate(3, &buffer) == null);
}

test {
    std.testing.refAllDecls(@This());
}
