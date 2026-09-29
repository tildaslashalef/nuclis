//! Allocation-free pre-tokenization with the GPT-2 byte-level pattern, as a
//! Hugging Face `ByteLevel` pre-tokenizer with `use_regex` applies it:
//! `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`.
//! Contractions are case-sensitive, digit runs are unbounded, and marks are
//! neither letters nor numbers. Splits borrow the input bytes; the category
//! table is pre.zig's.
const std = @import("std");
const pre = @import("pre.zig");
const Char = pre.Char;

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
        if (first.cp == '\'') {
            for ([_][]const u8{ "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" }) |suffix| {
                if (std.mem.startsWith(u8, self.text[start..], suffix)) return self.finish(start, start + suffix.len);
            }
        }
        // One optional ASCII space, then a run of one class.
        const body = if (first.cp == ' ') start + 1 else start;
        const lead = self.at(body);
        const run: ?*const fn (Char) bool = if (lead.letter()) &Char.letter else if (lead.number()) &Char.number else if (other(lead)) &other else null;
        if (run) |inside| {
            var end = body;
            while (inside(self.at(end))) end += self.at(end).len;
            return self.finish(start, end);
        }
        // Whitespace: the run, less its last character when a non-space follows
        // (that one prefixes the next piece), but never less than one character.
        var end = start;
        var last = start;
        while (self.at(end).space()) {
            last = end;
            end += self.at(end).len;
        }
        if (end == start) return self.finish(start, start + first.len);
        if (end < self.text.len and last > start) return self.finish(start, last);
        return self.finish(start, end);
    }

    fn finish(self: *Iterator, start: usize, end: usize) []const u8 {
        self.position = end;
        return self.text[start..end];
    }
};

// Expected pieces are the reference pre-tokenizer's on the same strings.
test "gpt2 contractions, class runs, and whitespace lookahead" {
    const cases = .{
        .{ "don't DON'T we're I'M 's", &[_][]const u8{ "don", "'t", " DON", "'", "T", " we", "'re", " I", "'", "M", " '", "s" } },
        .{ "12345 3.14 ²٣x", &[_][]const u8{ "12345", " 3", ".", "14", " ²٣", "x" } },
        .{ "Grüße 世界 🙂 e\u{301}", &[_][]const u8{ "Grüße", " 世界", " 🙂", " e", "\u{301}" } },
        .{ "  hello  world  ", &[_][]const u8{ " ", " hello", " ", " world", "  " } },
        .{ "\t\ta \t\n x", &[_][]const u8{ "\t", "\t", "a", " \t\n", " x" } },
        .{ "a\n\nb\n", &[_][]const u8{ "a", "\n", "\n", "b", "\n" } },
        .{ "x\u{a0}y\u{3000}z", &[_][]const u8{ "x", "\u{a0}", "y", "\u{3000}", "z" } },
        .{ "fn(){ !?", &[_][]const u8{ "fn", "(){", " !?" } },
        .{ "\x1c\x1d\x1e\x1f\x0b\x0c", &[_][]const u8{ "\x1c\x1d\x1e\x1f", "\x0b\x0c" } },
        .{ "a\u{85}b\u{2028}c \u{200b}d", &[_][]const u8{ "a", "\u{85}", "b", "\u{2028}", "c", " \u{200b}", "d" } },
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
