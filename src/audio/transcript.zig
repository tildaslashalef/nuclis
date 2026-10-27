//! Transcripts of whole recordings, shared by `nuclis transcribe` and
//! `POST /v1/audio/transcriptions` so both give the same text: a `Run`
//! decodes and feeds a recording in pieces of `piece_samples`, a
//! `Segmenter` groups the transcriber's timed tokens into segments and
//! words as they become final, and `writeSrt`/`writeVtt` write subtitles.
//! Times are the rows that emitted the tokens (80 ms each), not a forced
//! alignment. docs/guide/api.md § Transcription (files).
const std = @import("std");
const inference = @import("inference");

const audio = inference.audio.audio;
const asr = inference.audio.nemotron_asr;
const rnnt = inference.audio.nemotron_rnnt;

/// One encoder row, the unit of every time here.
pub const row_ms = 80;
/// A pause between words that ends a segment: 1.04 s. At 0.8 s, 14 of the
/// speech corpus's 60 one-sentence clips split mid-sentence; at 1.04 s, 4
/// (docs/guide/api.md § Segments).
pub const pause_rows = 13;
/// How long a segment without terminal punctuation waits for a late
/// comma or period before it is final: 4 s (the corpus's latest was 2.6 s).
pub const settle_rows = 50;
/// A segment ends at the next word once it is this long: 30 s.
pub const max_segment_rows = 375;
/// Samples decoded and pushed per step: 2.56 s, 32 rows.
pub const piece_samples = 32 * 1280;
/// Encoded bytes `nuclis transcribe` reads, and the route's default bound.
pub const max_file_bytes: usize = 256 << 20;

pub const Word = struct {
    /// Owned by the segmenter; trimmed, never empty.
    word: []const u8,
    start_ms: u64,
    end_ms: u64,
};

pub const Segment = struct {
    id: usize,
    start_ms: u64,
    end_ms: u64,
    /// Owned by the segmenter; spaces collapsed and trimmed.
    text: []const u8,
    /// The locale of the language tag the model emitted in the segment,
    /// else the last one before it, else the prompt's; owned.
    language: ?[]const u8,
    /// `Segmenter.words[first_word..][0..words]`.
    first_word: usize,
    words: usize,
};

/// One token as the segmenter reads it.
pub const Piece = struct {
    row: u32,
    /// The text with the replacement as spaces; empty for a special token.
    text: []const u8,
    /// A language tag's locale.
    tag: ?[]const u8 = null,
};

/// Segments: a new one starts at a word after terminal punctuation, after
/// a pause of `pause_rows` or more since the last word's sound, or once the
/// open one reached `max_segment_rows`. Times follow the pieces that hold
/// a letter or digit: the model emits punctuation late, after the pause it
/// marks. A segment is final once the next one starts, at the end, or once
/// the decoder is past its last token by `pause_rows` (it ends a sentence)
/// or `settle_rows` (it does not).
pub const Segmenter = struct {
    gpa: std.mem.Allocator,
    /// Owns every segment's and word's text and language.
    arena: std.heap.ArenaAllocator,
    segments: std.ArrayList(Segment) = .empty,
    words: std.ArrayList(Word) = .empty,
    /// Tokens of the transcriber already taken.
    consumed: usize = 0,
    /// The language a segment without a tag of its own takes: the last tag
    /// seen, at first the prompt's locale (null under `auto`).
    language: ?[]const u8,
    open: ?Open = null,
    /// The open segment's text and its open word's, in `gpa`.
    text: std.ArrayList(u8) = .empty,
    word: std.ArrayList(u8) = .empty,
    scratch: std.ArrayList(u8) = .empty,

    const Open = struct {
        first_row: u32,
        /// The last token's row, and the last one with a letter or digit.
        last_row: u32,
        sound_row: u32,
        language: ?[]const u8 = null,
        first_word: usize,
        /// The text ends a sentence.
        closed: bool = false,
        /// A piece of spaces alone came since the last word piece.
        space: bool = false,
        word_open: bool = false,
        word_row: u32 = 0,
        word_last: u32 = 0,
    };

    /// `language`: the prompt's locale when it is not `auto`; borrowed.
    pub fn init(gpa: std.mem.Allocator, language: ?[]const u8) Segmenter {
        return .{ .gpa = gpa, .arena = .init(gpa), .language = language };
    }

    pub fn deinit(self: *Segmenter) void {
        self.segments.deinit(self.gpa);
        self.words.deinit(self.gpa);
        self.text.deinit(self.gpa);
        self.word.deinit(self.gpa);
        self.scratch.deinit(self.gpa);
        self.arena.deinit();
        self.* = undefined;
    }

    /// Takes the transcriber's tokens not yet taken; `rows` is how many
    /// rows it decoded, `final` that the stream ended. Returns how many
    /// segments became final.
    pub fn update(self: *Segmenter, model: *const asr.Model, tokens: []const rnnt.Token, rows: usize, final: bool) !usize {
        const before = self.segments.items.len;
        for (tokens[self.consumed..]) |t| {
            if (model.table.isSpecial(t.id)) {
                const tag = if (t.id < model.tags.len) model.tags[t.id] else null;
                try self.take(.{ .row = t.row, .text = "", .tag = tag });
                continue;
            }
            // The piece with the replacement as spaces, as the table writes it.
            self.scratch.clearRetainingCapacity();
            var rest = model.table.piece(t.id) orelse "";
            while (std.mem.indexOf(u8, rest, model.table.replacement)) |at| {
                try self.scratch.appendSlice(self.gpa, rest[0..at]);
                try self.scratch.append(self.gpa, ' ');
                rest = rest[at + model.table.replacement.len ..];
            }
            try self.scratch.appendSlice(self.gpa, rest);
            try self.take(.{ .row = t.row, .text = self.scratch.items });
        }
        self.consumed = tokens.len;
        try self.settle(rows, final);
        return self.segments.items.len - before;
    }

    /// One token. Tags and other special tokens carry no text.
    pub fn take(self: *Segmenter, piece: Piece) !void {
        if (piece.text.len == 0) {
            if (piece.tag) |tag| {
                if (self.open) |*o| o.language = tag;
                self.language = tag;
            }
            return;
        }
        const blank = std.mem.trim(u8, piece.text, " ").len == 0;
        if (blank) {
            if (self.open) |*o| o.space = true;
            return;
        }
        const starts = piece.text[0] == ' ' or (if (self.open) |o| o.space else false);
        if (self.open) |o| if (starts and (o.closed or piece.row >= o.sound_row + pause_rows or piece.row >= o.first_row + max_segment_rows))
            try self.close();
        if (self.open == null) self.open = .{ .first_row = piece.row, .last_row = piece.row, .sound_row = piece.row, .first_word = self.words.items.len };
        const o = &self.open.?;
        if (starts or !o.word_open) {
            try self.closeWord();
            o.word_open = true;
            o.word_row = piece.row;
        }
        try appendCollapsed(self.gpa, &self.text, piece.text, starts);
        try appendCollapsed(self.gpa, &self.word, piece.text, false);
        o.last_row = piece.row;
        if (sounds(piece.text)) {
            o.sound_row = piece.row;
            o.word_last = piece.row;
        }
        o.space = false;
        o.closed = endsSentence(self.text.items);
    }

    /// Closes the open segment once no later token can join it, or at the end.
    pub fn settle(self: *Segmenter, rows: usize, final: bool) !void {
        const o = self.open orelse return;
        const wait: usize = if (o.closed) pause_rows else settle_rows;
        if (final or rows >= @as(usize, o.last_row) + wait) try self.close();
    }

    fn closeWord(self: *Segmenter) !void {
        const o = &self.open.?;
        if (!o.word_open) return;
        o.word_open = false;
        defer self.word.clearRetainingCapacity();
        const text = std.mem.trim(u8, self.word.items, " ");
        if (text.len == 0) return;
        try self.words.append(self.gpa, .{
            .word = try self.arena.allocator().dupe(u8, text),
            .start_ms = @as(u64, o.word_row) * row_ms,
            .end_ms = (@as(u64, o.word_last) + 1) * row_ms,
        });
    }

    fn close(self: *Segmenter) !void {
        try self.closeWord();
        const o = self.open.?;
        defer {
            self.open = null;
            self.text.clearRetainingCapacity();
        }
        const arena = self.arena.allocator();
        const language = o.language orelse self.language;
        try self.segments.append(self.gpa, .{
            .id = self.segments.items.len,
            .start_ms = @as(u64, o.first_row) * row_ms,
            .end_ms = (@as(u64, o.sound_row) + 1) * row_ms,
            .text = try arena.dupe(u8, std.mem.trim(u8, self.text.items, " ")),
            .language = if (language) |l| try arena.dupe(u8, l) else null,
            .first_word = o.first_word,
            .words = self.words.items.len - o.first_word,
        });
    }

    pub fn wordsOf(self: *const Segmenter, segment: Segment) []const Word {
        return self.words.items[segment.first_word..][0..segment.words];
    }

    /// The final segments' texts joined by spaces: the transcript.
    pub fn writeText(self: *const Segmenter, out: *std.Io.Writer) std.Io.Writer.Error!void {
        for (self.segments.items, 0..) |s, i| {
            if (i > 0) try out.writeByte(' ');
            try out.writeAll(s.text);
        }
    }
};

/// `text` appended with every run of spaces as one and none at the start;
/// `space` puts one before it.
fn appendCollapsed(gpa: std.mem.Allocator, out: *std.ArrayList(u8), text: []const u8, space: bool) !void {
    if (space and out.items.len > 0 and out.items[out.items.len - 1] != ' ') try out.append(gpa, ' ');
    for (text) |c| {
        if (c == ' ' and (out.items.len == 0 or out.items[out.items.len - 1] == ' ')) continue;
        try out.append(gpa, c);
    }
}

/// Whether a piece holds anything but spaces and punctuation: a letter or
/// a digit, of any script.
fn sounds(text: []const u8) bool {
    var it = (std.unicode.Utf8View.init(text) catch return true).iterator();
    while (it.nextCodepoint()) |c| {
        if (c < 0x80) {
            if (std.ascii.isAlphanumeric(@intCast(c))) return true;
            continue;
        }
        const punctuation = c == 0xA1 or c == 0xAB or c == 0xBB or c == 0xBF or
            (c >= 0x2000 and c <= 0x206F) or (c >= 0x3000 and c <= 0x303F) or (c >= 0xFF01 and c <= 0xFF0F);
        if (!punctuation) return true;
    }
    return false;
}

/// Whether `text` ends a sentence: terminal punctuation, then perhaps
/// closing quotes or brackets.
fn endsSentence(text: []const u8) bool {
    const closers = [_][]const u8{ "\"", "'", ")", "]", "”", "’", "»" };
    const terminals = [_][]const u8{ ".", "?", "!", "…", "。", "？", "！" };
    var rest = std.mem.trimEnd(u8, text, " ");
    strip: while (true) {
        for (closers) |c| if (std.mem.endsWith(u8, rest, c)) {
            rest = rest[0 .. rest.len - c.len];
            continue :strip;
        };
        break;
    }
    for (terminals) |t| if (std.mem.endsWith(u8, rest, t)) return true;
    return false;
}

/// A recording fed to a transcriber a piece at a time, each piece
/// segmented as it is decoded. Borrows the reader, the transcriber, and
/// the segmenter; owns the piece buffer.
pub const Run = struct {
    reader: *audio.Reader,
    transcriber: *asr.Transcriber,
    segmenter: *Segmenter,
    samples: []f32,
    /// Samples pushed so far.
    fed: u64 = 0,
    finished: bool = false,

    pub fn init(gpa: std.mem.Allocator, reader: *audio.Reader, transcriber: *asr.Transcriber, segmenter: *Segmenter) !Run {
        return .{ .reader = reader, .transcriber = transcriber, .segmenter = segmenter, .samples = try gpa.alloc(f32, piece_samples) };
    }

    pub fn deinit(self: *Run, gpa: std.mem.Allocator) void {
        gpa.free(self.samples);
        self.* = undefined;
    }

    /// Decodes and pushes the next piece, finishing the stream at the end
    /// of the audio; returns how many segments became final.
    /// `UnsupportedAudioFormat` when the file stops decoding midway.
    pub fn step(self: *Run) !usize {
        std.debug.assert(!self.finished);
        const n = try self.reader.read(self.samples);
        if (n > 0) _ = try self.transcriber.push(self.samples[0..n]);
        self.fed += n;
        if (n < self.samples.len) {
            _ = try self.transcriber.finish();
            self.finished = true;
        }
        return self.segmenter.update(self.transcriber.model, self.transcriber.tokens.items, self.transcriber.rowsDecoded(), self.finished);
    }

    /// The audio pushed so far, in seconds.
    pub fn seconds(self: *const Run) f64 {
        return @as(f64, @floatFromInt(self.fed)) / audio.sample_rate;
    }
};

pub fn seconds(ms: u64) f64 {
    return @as(f64, @floatFromInt(ms)) / 1000;
}

/// SubRip: numbered from 1, times `HH:MM:SS,mmm`.
pub fn writeSrt(out: *std.Io.Writer, segments: []const Segment) std.Io.Writer.Error!void {
    for (segments, 1..) |s, n| {
        try out.print("{d}\n", .{n});
        try writeCue(out, s, ',');
    }
}

/// WebVTT: the header, then cues with times `HH:MM:SS.mmm`.
pub fn writeVtt(out: *std.Io.Writer, segments: []const Segment) std.Io.Writer.Error!void {
    try out.writeAll("WEBVTT\n\n");
    for (segments) |s| try writeCue(out, s, '.');
}

fn writeCue(out: *std.Io.Writer, s: Segment, separator: u8) std.Io.Writer.Error!void {
    try writeTime(out, s.start_ms, separator);
    try out.writeAll(" --> ");
    try writeTime(out, s.end_ms, separator);
    // A blank line ends a cue, so the text never holds one.
    try out.print("\n{s}\n\n", .{s.text});
}

pub fn writeTime(out: *std.Io.Writer, ms: u64, separator: u8) std.Io.Writer.Error!void {
    try out.print("{d:0>2}:{d:0>2}:{d:0>2}{c}{d:0>3}", .{ ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, separator, ms % 1000 });
}

/// A segment as JSON fields of the object `w` has open: `id`, `start`,
/// `end` (seconds), `text`, `language`, and `words` when asked.
pub fn writeSegmentFields(w: *std.json.Stringify, segmenter: *const Segmenter, s: Segment, words: bool) std.Io.Writer.Error!void {
    try w.objectField("id");
    try w.write(s.id);
    try w.objectField("start");
    try w.write(seconds(s.start_ms));
    try w.objectField("end");
    try w.write(seconds(s.end_ms));
    try w.objectField("text");
    try w.write(s.text);
    try w.objectField("language");
    try w.write(s.language);
    if (words) {
        try w.objectField("words");
        try writeWords(w, segmenter.wordsOf(s));
    }
}

/// `[{word, start, end}, …]`.
pub fn writeWords(w: *std.json.Stringify, words: []const Word) std.Io.Writer.Error!void {
    try w.beginArray();
    for (words) |word| {
        try w.beginObject();
        try w.objectField("word");
        try w.write(word.word);
        try w.objectField("start");
        try w.write(seconds(word.start_ms));
        try w.objectField("end");
        try w.write(seconds(word.end_ms));
        try w.endObject();
    }
    try w.endArray();
}

// ----- tests: pieces as the model emits them, no model file -----

fn feed(s: *Segmenter, pieces: []const Piece) !void {
    for (pieces) |p| try s.take(p);
}

test "segments end at terminal punctuation, a pause, or the length bound; tags name them" {
    var s: Segmenter = .init(std.testing.allocator, null);
    defer s.deinit();
    // "Hello world. How are" + a pause + "you?" then a tag.
    try feed(&s, &.{
        .{ .row = 2, .text = "Hel" },
        .{ .row = 3, .text = "lo" },
        .{ .row = 5, .text = " world" },
        .{ .row = 6, .text = "." },
        .{ .row = 6, .text = " " },
        .{ .row = 6, .text = "", .tag = "en-US" },
        .{ .row = 9, .text = " How" },
        .{ .row = 10, .text = " are" },
        .{ .row = 23, .text = " you" },
        .{ .row = 24, .text = "?\"" },
        .{ .row = 25, .text = " " },
        .{ .row = 25, .text = "", .tag = "de-DE" },
    });
    try std.testing.expectEqual(@as(usize, 2), s.segments.items.len);
    const first = s.segments.items[0];
    try std.testing.expectEqualStrings("Hello world.", first.text);
    try std.testing.expectEqualStrings("en-US", first.language.?);
    try std.testing.expectEqual(@as(u64, 160), first.start_ms);
    // The period came with the last word's row; a late one would not move the end.
    try std.testing.expectEqual(@as(u64, 480), first.end_ms);
    const words = s.wordsOf(first);
    try std.testing.expectEqual(@as(usize, 2), words.len);
    try std.testing.expectEqualStrings("Hello", words[0].word);
    try std.testing.expectEqual(@as(u64, 160), words[0].start_ms);
    try std.testing.expectEqual(@as(u64, 320), words[0].end_ms);
    try std.testing.expectEqualStrings("world.", words[1].word);
    try std.testing.expectEqual(@as(u64, 480), words[1].end_ms);
    // The pause split "How are" from "you?", which carries no tag of its own yet.
    try std.testing.expectEqualStrings("How are", s.segments.items[1].text);
    try std.testing.expectEqualStrings("en-US", s.segments.items[1].language.?);
    try s.settle(36, false);
    try std.testing.expectEqual(@as(usize, 2), s.segments.items.len);
    try s.settle(37, false);
    try std.testing.expectEqual(@as(usize, 3), s.segments.items.len);
    try std.testing.expectEqualStrings("you?\"", s.segments.items[2].text);
    try std.testing.expectEqualStrings("de-DE", s.segments.items[2].language.?);
    try std.testing.expectEqual(@as(usize, 2), s.segments.items[2].id);

    var text: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer text.deinit();
    try s.writeText(&text.writer);
    try std.testing.expectEqualStrings("Hello world. How are you?\"", text.written());

    // A late period joins its sentence, which waits for it before it is final.
    var late: Segmenter = .init(std.testing.allocator, null);
    defer late.deinit();
    try feed(&late, &.{ .{ .row = 0, .text = " Fin" }, .{ .row = 1, .text = "al" } });
    try late.settle(1 + pause_rows, false);
    try std.testing.expectEqual(@as(usize, 0), late.segments.items.len);
    try feed(&late, &.{ .{ .row = 30, .text = "…" }, .{ .row = 30, .text = " " } });
    try std.testing.expect(late.open.?.closed);
    try late.settle(30 + pause_rows, false);
    try std.testing.expectEqualStrings("Final…", late.segments.items[0].text);
    try std.testing.expectEqual(@as(u64, 160), late.segments.items[0].end_ms);
    try std.testing.expect(sounds("é") and sounds("в") and sounds("7") and !sounds("。") and !sounds("”.") and !sounds("¿"));

    // A decimal point continues a sentence; a pause inside a word does not split it.
    var d: Segmenter = .init(std.testing.allocator, "fr");
    defer d.deinit();
    try feed(&d, &.{ .{ .row = 0, .text = " 3" }, .{ .row = 1, .text = "." }, .{ .row = 2, .text = "5" }, .{ .row = 40, .text = "0" }, .{ .row = 41, .text = " ok" } });
    try d.settle(0, true);
    try std.testing.expectEqual(@as(usize, 1), d.segments.items.len);
    try std.testing.expectEqualStrings("3.50 ok", d.segments.items[0].text);
    try std.testing.expectEqualStrings("fr", d.segments.items[0].language.?);
    try std.testing.expectEqualStrings("3.50", d.wordsOf(d.segments.items[0])[0].word);

    // Unbroken speech is cut at the next word after 30 s.
    var long: Segmenter = .init(std.testing.allocator, null);
    defer long.deinit();
    var row: u32 = 0;
    while (row < 400) : (row += 5) try long.take(.{ .row = row, .text = " la" });
    try long.settle(row, true);
    try std.testing.expectEqual(@as(usize, 2), long.segments.items.len);
    try std.testing.expectEqual(@as(u64, max_segment_rows * row_ms), long.segments.items[1].start_ms);
    try std.testing.expect(long.segments.items[0].language == null);
}

test "words: a leading space starts one, punctuation joins it, times span its sounding rows" {
    var s: Segmenter = .init(std.testing.allocator, null);
    defer s.deinit();
    try feed(&s, &.{
        .{ .row = 3, .text = "He" }, .{ .row = 4, .text = "llo" }, .{ .row = 4, .text = "," },
        .{ .row = 6, .text = " " },  .{ .row = 7, .text = "wor" }, .{ .row = 9, .text = " ld" },
        .{ .row = 10, .text = "." },
    });
    try s.settle(0, true);
    const words = s.words.items;
    try std.testing.expectEqual(@as(usize, 3), words.len);
    try std.testing.expectEqualStrings("Hello,", words[0].word);
    try std.testing.expectEqual(@as(u64, 240), words[0].start_ms);
    try std.testing.expectEqual(@as(u64, 400), words[0].end_ms);
    // A bare space starts the word; its text, and its time, begin after.
    try std.testing.expectEqualStrings("wor", words[1].word);
    try std.testing.expectEqual(@as(u64, 560), words[1].start_ms);
    // The period, emitted a row late, does not move the end.
    try std.testing.expectEqualStrings("ld.", words[2].word);
    try std.testing.expectEqual(@as(u64, 800), words[2].end_ms);
}

test "srt and vtt: numbered cues, times by separator, the WEBVTT header" {
    const segments = [_]Segment{
        .{ .id = 0, .start_ms = 0, .end_ms = 1520, .text = "Hello world.", .language = "en-US", .first_word = 0, .words = 0 },
        .{ .id = 1, .start_ms = 3_725_080, .end_ms = 3_727_000, .text = "Später.", .language = null, .first_word = 0, .words = 0 },
    };
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try writeSrt(&out.writer, &segments);
    try std.testing.expectEqualStrings("1\n00:00:00,000 --> 00:00:01,520\nHello world.\n\n2\n01:02:05,080 --> 01:02:07,000\nSpäter.\n\n", out.written());
    out.clearRetainingCapacity();
    try writeVtt(&out.writer, &segments);
    try std.testing.expectEqualStrings("WEBVTT\n\n00:00:00.000 --> 00:00:01.520\nHello world.\n\n01:02:05.080 --> 01:02:07.000\nSpäter.\n\n", out.written());
    out.clearRetainingCapacity();
    try writeVtt(&out.writer, &.{});
    try std.testing.expectEqualStrings("WEBVTT\n\n", out.written());
}

test "a segment as JSON, with and without its words" {
    var s: Segmenter = .init(std.testing.allocator, null);
    defer s.deinit();
    try feed(&s, &.{ .{ .row = 1, .text = " J" }, .{ .row = 3, .text = "a" }, .{ .row = 5, .text = "." } });
    try s.settle(0, true);
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    var w: std.json.Stringify = .{ .writer = &out.writer };
    try w.beginObject();
    try writeSegmentFields(&w, &s, s.segments.items[0], true);
    try w.endObject();
    try std.testing.expectEqualStrings("{\"id\":0,\"start\":0.08,\"end\":0.32,\"text\":\"Ja.\",\"language\":null,\"words\":[{\"word\":\"Ja.\",\"start\":0.08,\"end\":0.32}]}", out.written());
}
