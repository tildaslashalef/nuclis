//! Allocation-free pre-tokenization for `pre = "llama4"` vocabularies (the
//! gpt-4o pattern), matching the pinned reference's realization of that
//! regex, not its Unicode original: the reference folds every letter class
//! into one, so "uppercase" is a letter that is not ASCII a–z, "lowercase" a
//! letter that is not ASCII A–Z, and combining marks are not letters at all
//! (docs/reference/muse-glimmer.md § Tokenizer). Splits borrow the input
//! bytes; the category table is pre.zig's.
const std = @import("std");
const pre = @import("pre.zig");
const Char = pre.Char;

fn upperish(ch: Char) bool {
    return ch.letter() and !(ch.cp >= 'a' and ch.cp <= 'z');
}
fn lowerish(ch: Char) bool {
    return ch.letter() and !(ch.cp >= 'A' and ch.cp <= 'Z');
}
/// The `[^\s\p{L}\p{N}]` class: marks, punctuation, symbols, controls.
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
        // A word: one optional prefix that is not a letter, number, or CR/LF
        // (whitespace and marks included), the letter run, a contraction.
        const word = if (!first.newline() and !first.letter() and !first.number()) start + first.len else start;
        if (self.at(word).letter()) {
            var end = self.wordEnd(word);
            if (self.at(end).cp == '\'') {
                for ([_][]const u8{ "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" }) |suffix| {
                    if (self.text.len - end >= suffix.len and std.ascii.eqlIgnoreCase(self.text[end..][0..suffix.len], suffix)) {
                        end += suffix.len;
                        break;
                    }
                }
            }
            return self.finish(start, end);
        }
        if (first.number()) {
            var end = start;
            var count: usize = 0;
            while (count < 3 and self.at(end).number()) : (count += 1) end += self.at(end).len;
            return self.finish(start, end);
        }
        const punct = if (first.cp == ' ') start + first.len else start;
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
        return self.finish(start, start + first.len);
    }

    /// Where the letter run starting at `start` ends under the reference's
    /// ordered alternatives: greedy upper-ish letters then at least one
    /// lower-ish one; failing that, the run is cut after its last letter that
    /// is both (a non-ASCII letter); failing that, the whole ASCII-uppercase run.
    fn wordEnd(self: Iterator, start: usize) usize {
        var end = start;
        var seam: ?usize = null;
        while (upperish(self.at(end))) {
            const ch = self.at(end);
            end += ch.len;
            if (lowerish(ch)) seam = end;
        }
        if (lowerish(self.at(end))) {
            while (lowerish(self.at(end))) end += self.at(end).len;
            return end;
        }
        return seam orelse end;
    }

    fn finish(self: *Iterator, start: usize, end: usize) []const u8 {
        self.position = end;
        return self.text[start..end];
    }
};

// Expected pieces come from the pinned reference's `unicode_regex_split` run
// on the same strings (docs/reference/muse-glimmer.md § Tokenizer).
test "gpt4o case seams, contractions, digit triples, and trailing slashes" {
    const cases = .{
        .{ "HelloWORLD helloWORLD", &[_][]const u8{ "Hello", "WORLD", " hello", "WORLD" } },
        .{ "ÀBC ABCÀÉ ÀÉBC ÀÉbcDE", &[_][]const u8{ "À", "BC", " ABCÀÉ", " ÀÉ", "BC", " ÀÉbc", "DE" } },
        .{ "don't I'M we're x'S y'LL don'tt 's", &[_][]const u8{ "don't", " I'M", " we're", " x'S", " y'LL", " don't", "t", " '", "s" } },
        .{ "e\u{301} \u{301}z", &[_][]const u8{ "e", "\u{301}", " \u{301}", "z" } },
        .{ "1234567 ²٣ abc123def 3.14", &[_][]const u8{ "123", "456", "7", " ", "²٣", " abc", "123", "def", " ", "3", ".", "14" } },
        .{ "a/b//\n!!\r\n/x a//", &[_][]const u8{ "a", "/b", "//\n", "!!\r\n/", "x", " a", "//" } },
        .{ "\t  a\r\nb\n", &[_][]const u8{ "\t ", " a", "\r\n", "b", "\n" } },
        .{ "  hello  world  ", &[_][]const u8{ " ", " hello", " ", " world", "  " } },
        .{ "Grüße 世界 🙂 é", &[_][]const u8{ "Grüße", " 世界", " 🙂", " é" } },
        .{ "x\u{a0}y\u{2003}z\u{200b}w\x01v", &[_][]const u8{ "x", "\u{a0}y", "\u{2003}z", "\u{200b}w", "\x01v" } },
        .{ "\n\t \n  x \n", &[_][]const u8{ "\n\t \n", " ", " x", " \n" } },
        .{ "ok?!\n\n\nz \r\n/", &[_][]const u8{ "ok", "?!\n\n\n", "z", " \r\n", "/" } },
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
