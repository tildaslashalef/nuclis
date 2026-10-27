//! Allocation-free pre-tokenization with the case-seam pattern that
//! `transformers` 4.57.3 installs on a byte-level tokenizer loaded with
//! `fix_mistral_regex=True` (Qwen3-TTS's processor, docs/models/qwen3-tts.md
//! § Tokenizer and the rendered text):
//! `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|`
//! `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|`
//! `\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`.
//! Unlike gpt4o.zig there is no contraction suffix, digits are single, and
//! combining marks belong to both letter classes. The category table
//! (pre.zig's) has no case, so "upper" is a letter that is not ASCII a–z and
//! "lower" one that is not ASCII A–Z: exact for ASCII and for caseless
//! letters (CJK, Arabic, …); a word whose non-ASCII capital is followed by
//! ASCII capitals (`ÉCOLE`) splits after it, where the reference keeps it whole.
const std = @import("std");
const pre = @import("pre.zig");
const Char = pre.Char;

fn upper(ch: Char) bool {
    return ch.mark() or (ch.letter() and !(ch.cp >= 'a' and ch.cp <= 'z'));
}
fn lower(ch: Char) bool {
    return ch.mark() or (ch.letter() and !(ch.cp >= 'A' and ch.cp <= 'Z'));
}
/// `[^\r\n\p{L}\p{N}]`: the optional character before a word.
fn prefix(ch: Char) bool {
    return ch.len != 0 and !ch.newline() and !ch.letter() and !ch.number();
}
/// `[^\s\p{L}\p{N}]`: marks, punctuation, symbols, controls.
fn other(ch: Char) bool {
    return ch.len != 0 and ch.bits & 0x26 == 0;
}

pub const Iterator = struct {
    text: []const u8,
    position: usize = 0,

    pub fn init(text: []const u8) error{InvalidUtf8}!Iterator {
        if (!std.unicode.utf8ValidateSlice(text)) return error.InvalidUtf8;
        return .{ .text = text };
    }

    fn at(self: Iterator, offset: usize) Char {
        return pre.charAt(self.text, offset);
    }

    pub fn next(self: *Iterator) ?[]const u8 {
        if (self.position == self.text.len) return null;
        const start = self.position;
        const first = self.at(start);
        // Each word alternative with its optional prefix taken, then not
        // (which matters only when the prefix is a mark, itself a word character).
        const after = start + first.len;
        const with_prefix = prefix(first);
        if (with_prefix) if (self.upperLower(after)) |end| return self.finish(start, end);
        if (self.upperLower(start)) |end| return self.finish(start, end);
        if (with_prefix) if (self.capitals(after)) |end| return self.finish(start, end);
        if (self.capitals(start)) |end| return self.finish(start, end);
        if (first.number()) return self.finish(start, after);
        const punct = if (first.cp == ' ') after else start;
        if (other(self.at(punct))) {
            var end = punct;
            while (other(self.at(end))) end += self.at(end).len;
            while (self.at(end).newline() or self.at(end).cp == '/') end += self.at(end).len;
            return self.finish(start, end);
        }
        var end = start;
        var last_newline: usize = start;
        var last_space: usize = start;
        while (self.at(end).space()) {
            last_space = end;
            const ch = self.at(end);
            end += ch.len;
            if (ch.newline()) last_newline = end;
        }
        if (last_newline > start) return self.finish(start, last_newline);
        // Leave one whitespace for the next word's optional prefix.
        if (last_space > start and end < self.text.len) return self.finish(start, last_space);
        if (end > start) return self.finish(start, end);
        return self.finish(start, after);
    }

    /// `U*L+` at `p` as a backtracking matcher takes it: the greedy upper
    /// run, then the lower run that follows it; failing that, the run cut
    /// after its last character of both classes; failing that, no match.
    fn upperLower(self: Iterator, p: usize) ?usize {
        var end = p;
        var seam: ?usize = null;
        while (upper(self.at(end))) {
            const ch = self.at(end);
            end += ch.len;
            if (lower(ch)) seam = end;
        }
        if (lower(self.at(end))) {
            while (lower(self.at(end))) end += self.at(end).len;
            return end;
        }
        return seam;
    }

    /// `U+L*` at `p`.
    fn capitals(self: Iterator, p: usize) ?usize {
        var end = p;
        while (upper(self.at(end))) end += self.at(end).len;
        if (end == p) return null;
        while (lower(self.at(end))) end += self.at(end).len;
        return end;
    }

    fn finish(self: *Iterator, start: usize, end: usize) []const u8 {
        self.position = end;
        return self.text[start..end];
    }
};

// Expected pieces are the pinned `transformers` 4.57.3 `Split` pre-tokenizer's
// on the same strings, except where marked.
test "case seams without contractions, single digits, marks in words" {
    const cases = .{
        .{ "don't I'M we're McDonald's iPhone HTTPServer", &[_][]const u8{ "don", "'t", " I", "'M", " we", "'re", " Mc", "Donald", "'s", " i", "Phone", " HTTPServer" } },
        .{ "HelloWORLD helloWORLD", &[_][]const u8{ "Hello", "WORLD", " hello", "WORLD" } },
        .{ "Grüße 世界 🙂 é", &[_][]const u8{ "Grüße", " 世界", " 🙂", " é" } },
        .{ "1234567 ²٣ abc123def 3.14", &[_][]const u8{ "1", "2", "3", "4", "5", "6", "7", " ", "²", "٣", " abc", "1", "2", "3", "def", " ", "3", ".", "1", "4" } },
        .{ "a/b//\n!!\r\n/x a//", &[_][]const u8{ "a", "/b", "//\n", "!!\r\n/", "x", " a", "//" } },
        .{ "\t  a\r\nb\n", &[_][]const u8{ "\t ", " a", "\r\n", "b", "\n" } },
        .{ "  hello  world  ", &[_][]const u8{ " ", " hello", " ", " world", "  " } },
        .{ "x\u{a0}y\u{2003}z\u{200b}w\x01v", &[_][]const u8{ "x", "\u{a0}y", "\u{2003}z", "\u{200b}w", "\x01v" } },
        .{ "École ÀÉbcDE Привет ПРИВЕТ", &[_][]const u8{ "École", " ÀÉbc", "DE", " Привет", " ПРИВЕТ" } },
        .{ "e\u{301} \u{301}z !\u{301}a \u{301}A", &[_][]const u8{ "e\u{301}", " \u{301}z", " !\u{301}", "a", " \u{301}", "A" } },
        .{ "\n\t \n  x \n", &[_][]const u8{ "\n\t \n", " ", " x", " \n" } },
        .{ "ok?!\n\n\nz \r\n/", &[_][]const u8{ "ok", "?!\n\n\n", "z", " \r\n", "/" } },
        // The reference keeps both whole (its À and É are capitals only).
        .{ "ÀBC ÉCOLE", &[_][]const u8{ "À", "BC", " É", "COLE" } },
    };
    inline for (cases) |case| {
        var it = try Iterator.init(case[0]);
        for (case[1]) |expected| try std.testing.expectEqualStrings(expected, it.next().?);
        try std.testing.expectEqual(@as(?[]const u8, null), it.next());
    }
    try std.testing.expectError(error.InvalidUtf8, Iterator.init("\xff"));
    var empty = try Iterator.init("");
    try std.testing.expect(empty.next() == null);
}
