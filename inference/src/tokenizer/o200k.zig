//! Allocation-free pre-tokenization for `pre = "inkling2"` vocabularies: the
//! o200k expression of the checkpoint's own `tokenizer.json`, with its full
//! Unicode classes (docs/engine/tokenizer.md § Full text encoding):
//! `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|`
//! `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|`
//! `\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`.
//! Unlike gpt4o.zig (the folded rewrite) case comes from case_table.zig, so
//! a titlecase or caseless letter and a combining mark sit in both runs. The
//! alternatives are tried in order with the backtracking a regex engine does.
//! Letter, mark, number, and space are pre.zig's classes; splits borrow the
//! input bytes.
const std = @import("std");
const pre = @import("pre.zig");
const case_table = @import("case_table.zig");
const Char = pre.Char;

/// `case_table.upper`, `case_table.lower`, or 0 for a code point of neither.
fn caseOf(cp: u21) u2 {
    var low: usize = 0;
    var high: usize = case_table.ranges.len;
    while (low < high) {
        const mid = low + (high - low) / 2;
        const range = case_table.ranges[mid];
        if (cp < range[0]) high = mid else if (cp > range[1]) low = mid + 1 else return @intCast(range[2]);
    }
    return 0;
}

/// `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]`.
fn upperRun(ch: Char) bool {
    return ch.mark() or (ch.letter() and caseOf(ch.cp) != case_table.lower);
}
/// `[\p{Ll}\p{Lm}\p{Lo}\p{M}]`.
fn lowerRun(ch: Char) bool {
    return ch.mark() or (ch.letter() and caseOf(ch.cp) != case_table.upper);
}
/// `[^\r\n\p{L}\p{N}]`: whitespace, marks, punctuation, symbols, controls.
fn prefix(ch: Char) bool {
    return ch.len != 0 and !ch.newline() and !ch.letter() and !ch.number();
}
/// `[^\s\p{L}\p{N}]`: marks, punctuation, symbols, controls.
fn other(ch: Char) bool {
    return ch.len != 0 and !ch.space() and !ch.letter() and !ch.number();
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
        // The two word alternatives, each first with its optional prefix
        // character and then without it.
        const after_prefix: ?usize = if (prefix(first)) start + first.len else null;
        inline for (.{ caseSeam, upperWord }) |word| {
            if (after_prefix) |p| if (word(self.*, p)) |end| return self.finish(start, self.contraction(end));
            if (word(self.*, start)) |end| return self.finish(start, self.contraction(end));
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
        // `\s*[\r\n]+` ends after the run's last newline; `\s+(?!\S)` leaves
        // the run's last whitespace to the next word; `\s+` takes the rest.
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
        if (last_space > start and end < self.text.len) return self.finish(start, last_space);
        if (end > start) return self.finish(start, end);
        return self.finish(start, start + first.len);
    }

    /// `[Lu Lt Lm Lo M]*[Ll Lm Lo M]+` from `p`, or null. The greedy upper
    /// run gives characters back until one in the lower class starts the
    /// lower run: after the whole upper run if the next character is lower,
    /// else at the run's last character that is in both classes.
    fn caseSeam(self: Iterator, p: usize) ?usize {
        var end = p;
        var last_both: ?usize = null;
        while (upperRun(self.at(end))) {
            const ch = self.at(end);
            if (lowerRun(ch)) last_both = end;
            end += ch.len;
        }
        if (lowerRun(self.at(end))) {
            while (lowerRun(self.at(end))) end += self.at(end).len;
            return end;
        }
        // The lower run from the last shared character stops at the next
        // character, which is in the upper class only.
        const j = last_both orelse return null;
        return j + self.at(j).len;
    }

    /// `[Lu Lt Lm Lo M]+[Ll Lm Lo M]*` from `p`, or null.
    fn upperWord(self: Iterator, p: usize) ?usize {
        if (!upperRun(self.at(p))) return null;
        var end = p;
        while (upperRun(self.at(end))) end += self.at(end).len;
        while (lowerRun(self.at(end))) end += self.at(end).len;
        return end;
    }

    /// `(?i:'s|'t|'re|'ve|'m|'ll|'d)?` at `end`: the first suffix that
    /// matches under simple case folding (U+017F, long s, folds to `s`).
    fn contraction(self: Iterator, end: usize) usize {
        if (self.at(end).cp != '\'') return end;
        const rest = end + 1;
        for ([_][]const u8{ "s", "t", "re", "ve", "m", "ll", "d" }) |suffix| {
            var at_byte = rest;
            var matched = true;
            for (suffix) |letter| {
                const ch = self.at(at_byte);
                const folded = ch.len == 1 and std.ascii.toLower(@intCast(ch.cp)) == letter;
                if (!folded and !(letter == 's' and ch.cp == 0x17F)) {
                    matched = false;
                    break;
                }
                at_byte += ch.len;
            }
            if (matched) return at_byte;
        }
        return end;
    }

    fn finish(self: *Iterator, start: usize, end: usize) []const u8 {
        self.position = end;
        return self.text[start..end];
    }
};

fn expectPieces(text: []const u8, expected: []const []const u8) !void {
    var it = try Iterator.init(text);
    for (expected) |piece| try std.testing.expectEqualStrings(piece, it.next() orelse return error.TestExpectedEqual);
    try std.testing.expectEqual(@as(?[]const u8, null), it.next());
}

// Expected pieces are worked from the expression by hand.
test "o200k case seams: a capital run, then a lower run" {
    try expectPieces("HelloWORLD helloWORLD", &.{ "Hello", "WORLD", " hello", "WORLD" });
    try expectPieces("ÀBC ABCÀÉ ÀÉbcDE", &.{ "ÀBC", " ABCÀÉ", " ÀÉbc", "DE" });
    try expectPieces("ÉCOLE École éCOLE", &.{ "ÉCOLE", " École", " é", "COLE" });
    try expectPieces("CamelCaseHTTPServer", &.{ "Camel", "Case", "HTTPServer" });
}

test "o200k titlecase and modifier letters, caseless scripts" {
    // U+01C5 (Lt) is upper only; U+02B0 (Lm) and CJK (Lo) are in both runs.
    try expectPieces("\u{1C5}ab a\u{1C5}", &.{ "\u{1C5}ab", " a", "\u{1C5}" });
    try expectPieces("a\u{2B0}B x\u{2B0}", &.{ "a\u{2B0}", "B", " x\u{2B0}" });
    try expectPieces("世界Ab 世界AB", &.{ "世界Ab", " 世界", "AB" });
}

test "o200k combining marks belong to both runs and to the prefix class" {
    // A shared character ends the lower run when no lower letter follows the capitals.
    try expectPieces("e\u{301}cole E\u{301}COLE", &.{ "e\u{301}cole", " E\u{301}", "COLE" });
    try expectPieces(" \u{301}z", &.{" \u{301}z"});
    // A leading mark is the prefix of a lower run, and itself a run before a capital.
    try expectPieces("\u{301}z \u{301}A", &.{ "\u{301}z", " \u{301}", "A" });
}

test "o200k contractions, case-insensitive, after either word form" {
    try expectPieces("don't I'M we're x'S y'LL don'tt 's", &.{ "don't", " I'M", " we're", " x'S", " y'LL", " don't", "t", " '", "s" });
    try expectPieces("it'\u{17F} HE'Ve", &.{ "it'\u{17F}", " HE'Ve" });
}

test "o200k digit triples, punctuation with trailing newlines and slashes" {
    try expectPieces("1234567 ²٣ abc123def 3.14", &.{ "123", "456", "7", " ", "²٣", " abc", "123", "def", " ", "3", ".", "14" });
    try expectPieces("a/b//\n!!\r\n/x a//", &.{ "a", "/b", "//\n", "!!\r\n/", "x", " a", "//" });
}

test "o200k whitespace runs, CRLF, tabs, emoji" {
    try expectPieces("\t  a\r\nb\n", &.{ "\t ", " a", "\r\n", "b", "\n" });
    try expectPieces("  hello  world  ", &.{ " ", " hello", " ", " world", "  " });
    try expectPieces("\n\t \n  x \n", &.{ "\n\t \n", " ", " x", " \n" });
    try expectPieces("Grüße 世界 🙂 é 🙂🙂!", &.{ "Grüße", " 世界", " 🙂", " é", " 🙂🙂!" });
    try expectPieces("x\u{a0}y\u{2003}z\u{200b}w\x01v", &.{ "x", "\u{a0}y", "\u{2003}z", "\u{200b}w", "\x01v" });
}

test "o200k rejects invalid UTF-8 and ends an empty text at once" {
    try std.testing.expectError(error.InvalidUtf8, Iterator.init("\xff"));
    var empty = try Iterator.init("");
    try std.testing.expect(empty.next() == null);
}

test "the case table agrees with ASCII and holds the categories it names" {
    for (0..128) |c| {
        const want: u2 = if (c >= 'A' and c <= 'Z') case_table.upper else if (c >= 'a' and c <= 'z') case_table.lower else 0;
        try std.testing.expectEqual(want, caseOf(@intCast(c)));
    }
    try std.testing.expectEqual(@as(u2, case_table.upper), caseOf(0x1C5)); // Lt
    try std.testing.expectEqual(@as(u2, case_table.lower), caseOf(0xDF)); // ß, Ll
    try std.testing.expectEqual(@as(u2, 0), caseOf(0x2B0)); // Lm
    try std.testing.expectEqual(@as(u2, 0), caseOf(0x4E16)); // Lo
    var previous: u21 = 0;
    for (case_table.ranges, 0..) |range, i| {
        try std.testing.expect(range[0] <= range[1] and (i == 0 or range[0] > previous));
        previous = range[1];
    }
}
