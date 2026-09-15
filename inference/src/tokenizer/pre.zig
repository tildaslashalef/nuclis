//! Allocation-free qwen35 pre-tokenization of validated UTF-8 text.
//! Ordered alternatives match the pinned reference's custom splitter, including
//! combining marks and whitespace backtracking. Splits borrow the input bytes.
//! The splitting rules are those declared by the model's `qwen35` tokenizer;
//! the category table derives from Unicode data (see THIRD_PARTY_NOTICES.md).
//! Regenerate it with scripts/tokenizer-unicode.py; no host Unicode database is consulted.
const std = @import("std");
const ranges = @embedFile("unicode-ranges.bin"); // records: little-endian u32 start, u8 flags

fn flags(cp: u21) u8 {
    var low: usize = 0;
    var high: usize = ranges.len / 5;
    while (low + 1 < high) {
        const mid = low + (high - low) / 2;
        const start = std.mem.readInt(u32, ranges[mid * 5 ..][0..4], .little);
        if (start <= cp) low = mid else high = mid;
    }
    return ranges[low * 5 + 4];
}
const Char = struct {
    cp: u21 = 0,
    len: usize = 0,
    bits: u8 = 0,
    fn letter(self: Char) bool {
        return self.bits & 0x14 != 0;
    }
    fn number(self: Char) bool {
        return self.bits & 2 != 0;
    }
    fn space(self: Char) bool {
        return self.bits & 0x20 != 0;
    }
    fn newline(self: Char) bool {
        return self.cp == '\r' or self.cp == '\n';
    }
    fn punctuation(self: Char) bool {
        return self.len != 0 and self.bits == 0;
    }
};

pub const Iterator = struct {
    text: []const u8,
    position: usize = 0,

    pub fn init(text: []const u8) error{InvalidUtf8}!Iterator {
        if (!std.unicode.utf8ValidateSlice(text)) return error.InvalidUtf8;
        return .{ .text = text };
    }

    fn at(self: Iterator, offset: usize) Char {
        if (offset == self.text.len) return .{};
        const len = std.unicode.utf8ByteSequenceLength(self.text[offset]) catch unreachable;
        const cp = std.unicode.utf8Decode(self.text[offset..][0..len]) catch unreachable;
        return .{ .cp = cp, .len = len, .bits = flags(cp) };
    }

    pub fn next(self: *Iterator) ?[]const u8 {
        if (self.position == self.text.len) return null;
        const start = self.position;
        const first = self.at(start);
        var end = start + first.len;
        const second = self.at(end);
        // Case-insensitive contractions precede the optional-prefix letter run.
        if (first.cp == '\'') {
            for ([_][]const u8{ "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" }) |suffix| {
                if (self.text.len - start >= suffix.len and std.ascii.eqlIgnoreCase(self.text[start..][0..suffix.len], suffix))
                    return self.finish(start, start + suffix.len);
            }
        }
        if (!first.newline() and !first.number() and (first.letter() or second.letter())) {
            while (self.at(end).letter()) end += self.at(end).len;
            return self.finish(start, end);
        }
        if (first.number()) return self.finish(start, end);
        const punct = if (first.cp == ' ') second else first;
        if (punct.punctuation()) {
            end = if (first.cp == ' ') start + first.len else start;
            while (self.at(end).punctuation()) end += self.at(end).len;
            while (self.at(end).newline()) end += self.at(end).len;
            return self.finish(start, end);
        }
        end = start;
        var last_newline: usize = start;
        var last_space: usize = start;
        while (self.at(end).space()) {
            last_space = end;
            const ch = self.at(end);
            end += ch.len;
            if (ch.newline()) last_newline = end;
        }
        if (last_newline > start) return self.finish(start, last_newline);
        // Leave one whitespace for the next optional-prefix word alternative.
        if (last_space > start and end < self.text.len) return self.finish(start, last_space);
        if (end > start) return self.finish(start, end);
        return self.finish(start, start + first.len);
    }

    fn finish(self: *Iterator, start: usize, end: usize) []const u8 {
        self.position = end;
        return self.text[start..end];
    }
};

test "qwen35 ordered Unicode, contraction, digit, and whitespace splitting" {
    const cases = .{
        .{ "don't I'M we're", &[_][]const u8{ "don", "'t", " I", "'M", " we", "'re" } },
        .{ "123 ²٣", &[_][]const u8{ "1", "2", "3", " ", "²", "٣" } },
        .{ "Grüße 世界 🙂 e\u{301}", &[_][]const u8{ "Grüße", " 世界", " 🙂", " e\u{301}" } },
        .{ "\t  a\r\nb\n", &[_][]const u8{ "\t ", " a", "\r\n", "b", "\n" } },
        .{ "!\r\n  x  ", &[_][]const u8{ "!\r\n", " ", " x", "  " } },
        .{ "\u{a0}a\u{301} \u{301}z", &[_][]const u8{ "\u{a0}a\u{301}", " \u{301}z" } },
        .{ "\n\t \n  ", &[_][]const u8{ "\n\t \n", "  " } },
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

test "pinned Unicode table is ordered and spans supplementary characters" {
    try std.testing.expectEqual(@as(usize, 0), ranges.len % 5);
    var previous: u32 = 0;
    for (1..ranges.len / 5) |i| {
        const cp = std.mem.readInt(u32, ranges[i * 5 ..][0..4], .little);
        try std.testing.expect(cp > previous and cp <= 0x10ffff);
        previous = cp;
    }
    try std.testing.expect(flags(0x20000) & 4 != 0); // supplementary Han letter
    try std.testing.expect(flags(0x1d7ce) & 2 != 0); // mathematical digit
    try std.testing.expect(flags(0x301) & 0x10 != 0); // combining mark
    try std.testing.expect(flags(0x202f) & 0x20 != 0); // narrow no-break space
    try std.testing.expectEqual(@as(u8, 0), flags(0x1f642));
}
